import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import transformers
from dataclasses import dataclass, field
import sys
sys.path.append('/home/data1/zhangzr22/LLaVA_DATA/0717/LLaVA_agent_inference')
from typing import List, Tuple
import numpy as np
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
import re
from PIL import Image
import math
import cv2
from albumentations.pytorch.functional import img_to_tensor, mask_to_tensor
from torchvision.transforms import ToTensor, ToPILImage

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def save_tensor_image(tensor, filename, save_dir):
    to_pil = ToPILImage()
    img = to_pil(tensor.cpu())
    img.save(os.path.join(save_dir, filename))

def gray2rgb(gray):
    if gray.dim() == 3:  # Assuming gray has shape (C, H, W)
        gray = gray.unsqueeze(0)  # Shape becomes (1, C, H, W)
    elif gray.dim() == 2:  # Assuming gray has shape (H, W)
        gray = gray.unsqueeze(0).unsqueeze(0)  # Shape becomes (1, 1, H, W)
    rgb = gray.expand(-1, 3, -1, -1)  # Expand to (N, 3, H, W)
    return rgb


def direct_val(imgs):
    normalize = {"mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]}
    if len(imgs) != 1:
        pass
    imgs = img_to_tensor(imgs[0], normalize).unsqueeze(0)
    return imgs

class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        try:
            line = self.questions[index]
            image_file = line["image"]
            qs = line["text"]
            image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            if IMAGE_PLACEHOLDER in qs:
                if self.model_config.mm_use_im_start_end:
                    qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
                else:
                    qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
            else:
                if self.model_config.mm_use_im_start_end:
                    qs = image_token_se + "\n" + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
            image_tensor = process_images([image], self.image_processor, self.model_config)[0]

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

            if 'authentic' in image_file or '4cam_auth' in image_file:
                gt = np.zeros((336, 336), dtype=np.uint8)
                gt = mask_to_tensor(gt, num_classes=1, sigmoid=True)
            else:
                mask_image_file = image_file.replace('tampered', 'mask')
                gt = cv2.imread(os.path.join(self.image_folder, mask_image_file), cv2.IMREAD_GRAYSCALE)
                gt = cv2.resize(gt, (336, 336))
                gt = mask_to_tensor(gt, num_classes=1, sigmoid=True)
            gt = gray2rgb(gt)
            # 读取原始图像并转换为张量
            img = cv2.imread(os.path.join(self.image_folder, image_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (336, 336))
            img = img.reshape((-1, img.shape[-3], img.shape[-2], img.shape[-1]))
            img = direct_val(img).squeeze(0)

            return index, input_ids, image_tensor, gt, img
        except Exception as e:
            print(f"Error processing index {index}: {e}")
            print(f"Error data: {self.questions[index]}")

            next_index = index + 1
            if next_index >= len(self.questions):
                next_index = 1
            return self.__getitem__(next_index)

    def __len__(self):
        return len(self.questions)

@dataclass
class DataCollatorForVisualTextGeneration(object):
    tokenizer: transformers.PreTrainedTokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids] 
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=batch_first,
            padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        indices, input_ids, images, gts, imgs = zip(*batch)
        input_ids = self.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        images = torch.stack(images, dim=0)
        gts = torch.stack(gts, dim=0)
        imgs = torch.stack(imgs, dim=0)
        return indices, input_ids, images, gts, imgs


def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    collator = DataCollatorForVisualTextGeneration(tokenizer=tokenizer)
    data_loader = DataLoader(dataset, collate_fn=collator, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader

def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    model.config.tokenizer_padding_side = tokenizer.padding_side = "left"

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(
        questions,
        args.image_folder,
        tokenizer,
        image_processor,
        model.config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    breakpoint()
    for indices, input_ids, image_tensor, gts, imgs in tqdm(data_loader):
        with torch.inference_mode():
            image_tensor = image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True)
            gts = gts.to(dtype=torch.float16, device='cuda', non_blocking=True)
            imgs = imgs.to(dtype=torch.float16, device='cuda', non_blocking=True)
            
            pred_mask = model.agent_generate(imgs, gts)
            pred_mask = gray2rgb(pred_mask)
            
            output_ids = model.generate(
                input_ids.to(device='cuda', non_blocking=True),
                images=image_tensor,
                mask_images=pred_mask,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        for index, output in zip(indices, outputs):
            line = questions[index]
            idx = line["question_id"]
            cur_prompt = line["text"]
            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                       "prompt": cur_prompt,
                                       "text": output.strip(),
                                       "answer_id": ans_id,
                                       "model_id": model_name,
                                       "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default='/home/data1/zhangzr22/LLaVA_DATA/0717/LLaVA_agent/checkpoints/llava-pretrain-finetune-forgery-r16-lora')
    parser.add_argument("--model-base", type=str, default='/home/data1/zhangzr22/LLaVA_DATA/llava-v1.5-7b')
    # parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/home/data1/zhangzr22/LLaVA_DATA/sft_forgery_data/NIST")
    parser.add_argument("--question-file", type=str, default="/home/data1/zhangzr22/LLaVA_DATA/0717/LLaVA_agent_inference/llava/eval/table/forgery_question/question_forgery_NEW_COVER.jsonl")
    parser.add_argument("--answers-file", type=str, default="/home/data1/zhangzr22/LLaVA_DATA/0717/LLaVA_agent_inference/llava/eval/table/forgery_question/answer_forgery_NEW_COVER.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    eval_model(args)
