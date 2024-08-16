#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from functools import partial
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from ..location_model import LinearLayer,VisionTransformer
# from loss import FocalLoss, BinaryDiceLoss

class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        # 获取每个批次的大小 N
        N = targets.size()[0]
        # 平滑变量
        smooth = 1
        # 将宽高 reshape 到同一纬度
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)

        # 计算交集
        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)

        # 计算一个批次中平均每张图的损失
        loss = 1 - N_dice_eff.sum() / N

        return loss

def gray2rgb(gray):
    rgb = gray.expand(-1, 3, -1, -1)
    return rgb

def rgb2gray(rgb):
    b, g, r = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
    gray = 0.2989*r + 0.5870*g + 0.1140*b
    gray = torch.unsqueeze(gray, 1)
    return gray


import cv2
import re
import numpy as np
def normalize(pred, norm_method='minmax'):
    """
    [paper] Normalization Matters in Weakly Supervised Object Localization
    Args:
        pred: numpy.ndarray(size=(H, W), dtype=np.float)
    Returns:
        numpy.ndarray(size=(H, W), dtype=np.float) between 0 and 1.
        If input array is constant, a zero-array is returned.
    """
    pred = np.array(pred)
    min_value = float(re.findall('vmin[0-9.]{0,5}', norm_method)[0].split('vmin')[-1]) if 'vmin' in norm_method else pred.min()
    max_value = float(re.findall('vmax[0-9.]{0,5}', norm_method)[0].split('vmax')[-1]) if 'vmax' in norm_method else pred.max()

    if norm_method == 'minmax':
        pred -= min_value
        pred /= max_value
    elif norm_method == 'max':
        pred = np.maximum(0, pred)
        pred /= max_value
    elif norm_method == 'pas':
        percentile = 0.9
        pred -= min_value
        pred_copy = pred.flatten()
        pred_copy.sort()
        max_value = pred_copy[int(pred_copy.size * percentile)]
        pred /= max_value
        pred = np.minimum(1, pred)
    elif 'ivr' in norm_method:  # ivr0.45
        percentile = float(re.findall('ivr[0-9.]{0,5}', norm_method)[0].split('ivr')[-1])
        pred_copy = pred.flatten()
        pred_copy.sort()
        min_value = pred_copy[int(pred_copy.size * percentile)]
        pred -= min_value
        pred = np.maximum(0, pred)
        pred /= max_value
    else:
        raise NotImplementedError(f"Not Implemented norm_method: {norm_method}")
    return pred
    
class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size  # 32000
        ### Linear(in_features=4096, out_features=32000, bias=False)  llama的线性层
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # breakpoint()
        # Initialize weights and apply final processing
        self.post_init()

        self.pixel_model =  VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), img_size=336, num_classes=1)
        
        self.pixel_model_device = torch.device(f"cuda:7")
        ## 下面设置prompt
        ctx_vector_size = (5, 4096)
        ctx_vectors_pos = torch.empty(ctx_vector_size)
        nn.init.normal_(ctx_vectors_pos, std=0.02)
        self.ctx_token = nn.Parameter(ctx_vectors_pos) # 可优化的参数

    def get_model(self):
        return self.model

    def get_ctx_model(self):
        return self.ctx_token
    
    def get_pixel_model(self):
        return self.pixel_model
    
    @torch.no_grad()
    def agent_generate(self, img_pixel, mask_images_gt):
        pred_mask, _ = self.pixel_model(img_pixel, mask_images_gt)
        return pred_mask
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        mask_images: Optional[torch.FloatTensor] = None,
        mask_images_gt: Optional[torch.FloatTensor] = None,
        img_pixel: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        #
   
        if inputs_embeds is None:
            # breakpoint()
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                self.ctx_token,
                image_sizes
            )
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
    
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        mask_images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        # breakpoint()
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.inference_prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                mask_images,
                self.ctx_token,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        # breakpoint()
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
