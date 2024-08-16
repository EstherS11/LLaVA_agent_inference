import argparse
import torch
import sys
import numpy as np
sys.path.append('/home/data1/zhangzr22/LLaVA_DATA/0717/LLaVA_agent_inference')
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re
import cv2
from albumentations.pytorch.functional import img_to_tensor,mask_to_tensor
## 保存中间结果看一看
from torchvision.transforms import ToTensor, ToPILImage
import os
def save_tensor_image(tensor, filename, save_dir):
    to_pil = ToPILImage()
    img = to_pil(tensor.cpu())
    img.save(os.path.join(save_dir, filename))
def gray2rgb(gray):
    rgb = gray.expand(-1, 3, -1, -1)
    return rgb
def direct_val(imgs):
    normalize = {"mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]}
    if len(imgs) != 1:
        pass
    imgs = img_to_tensor(imgs[0], normalize).unsqueeze(0)
    return imgs
def gray2rgb(gray):
    rgb = gray.expand(-1, 3, -1, -1)
    return rgb
import debugpy
try:
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    debugpy.listen(("localhost", 9501))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out

def mask_image_parser(args):
    out = args.mask_image_file.split(args.sep)
    return out

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode
    # breakpoint()
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    image_files = image_parser(args)
    images = load_images(image_files)
    # breakpoint()
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    if args.mask_image_file is None:
        gt = np.zeros((336, 336), dtype=np.uint8)
        gt = mask_to_tensor(gt, num_classes=1, sigmoid=True).unsqueeze(0).to(model.device, dtype=torch.float16)
        gt = gray2rgb(gt)
    else:
        gt = cv2.imread(args.mask_image_file, cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, (336, 336))
        gt = mask_to_tensor(gt, num_classes=1, sigmoid= True).unsqueeze(0).to(model.device, dtype=torch.float16)
        gt = gray2rgb(gt)

    ## 1,3,336,336
    img = cv2.imread(args.image_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (336, 336))
    img = img.reshape((-1, img.shape[-3], img.shape[-2], img.shape[-1]))
    img = direct_val(img).to(model.device, dtype=torch.float16)
    
    with torch.inference_mode():
        pred_mask = model.agent_generate(img,gt)
        pred_mask = gray2rgb(pred_mask)
    if args.save_dir:
        save_tensor_image(pred_mask.squeeze(0), args.image_file.split('/')[-1], args.save_dir)
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    breakpoint()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            mask_images=pred_mask,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/data1/zhangzr22/LLaVA_DATA/llava-forgery-sft-0801-7b") #"/cache/llava_bbox_pretrain"
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default='/home/data1/zhangzr22/LLaVA_DATA/llava-forgery-sft-0801-7b/mask_visual')
    parser.add_argument("--image-file", type=str, default="/home/data1/zhangzr22/LLaVA_DATA/sft_forgery_data/NEW_COVER/authentic/86.jpg")
    parser.add_argument("--mask-image-file", type=str, default=None)
    parser.add_argument("--query", type=str, default='Is there any fogery in this image?')
    parser.add_argument("--conv-mode", type=str, default='v1')
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)
