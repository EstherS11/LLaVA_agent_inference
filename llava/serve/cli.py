import argparse
import torch
import numpy as np
import sys
sys.path.append('/home/data1/zhangzr22/LLaVA_DATA/0717/LLaVA_agent_inference')
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
import cv2
from albumentations.pytorch.functional import img_to_tensor,mask_to_tensor
## 保存中间结果看一看
from torchvision.transforms import ToTensor, ToPILImage
import os
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

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)


    model = model.to(dtype=torch.bfloat16)
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
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    image = load_image(args.image_file)
    image_size = image.size
    # Similar operation in model_worker.py
    image_tensor = process_images([image], image_processor, model.config)
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.bfloat16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.bfloat16)

    if args.mask_image_file is None:
        gt = np.zeros((336, 336), dtype=np.uint8)
        gt = mask_to_tensor(gt, num_classes=1, sigmoid=True).unsqueeze(0).to(model.device, dtype=torch.bfloat16)
        gt = gray2rgb(gt)
    else:
        gt = cv2.imread(args.mask_image_file, cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, (336, 336))
        gt = mask_to_tensor(gt, num_classes=1, sigmoid= True).unsqueeze(0).to(model.device, dtype=torch.bfloat16)
        gt = gray2rgb(gt)

    ## 1,3,336,336
    img = cv2.imread(args.image_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (336, 336))
    img = img.reshape((-1, img.shape[-3], img.shape[-2], img.shape[-1]))
    img = direct_val(img).to(model.device, dtype=torch.bfloat16)
    
    with torch.inference_mode():
        pred_mask = model.agent_generate(img,gt)
        pred_mask = gray2rgb(pred_mask)

    
    
    while True:
        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")

        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        # with torch.inference_mode():
        #     results = model(input_ids = input_ids, labels=input_ids, images = image_tensor)
        #     print(f"Loss: {results.loss}")
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                mask_images = pred_mask,
                image_sizes=[image_size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True)

        outputs = tokenizer.decode(output_ids[0]).strip()
        conv.messages[-1][-1] = outputs

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":  ## 
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model-path", type=str, default="/home/data1/zhangzr22/LLaVA_DATA/0717/LLaVA_agent/checkpoints/llava-v1.5-7b-fogery-lora-0801-gt/checkpoint-24000")
    # parser.add_argument("--model-base", type=str, default='/home/data1/zhangzr22/LLaVA_DATA/llava-forgery-sft-0729-7b')
    parser.add_argument("--model-path", type=str, default="/home/data1/zhangzr22/LLaVA_DATA/llava-forgery-sft-0801-7b")
    # parser.add_argument("--model-base", type=str, default='/home/ma-user/work/llava_ckpt/pretrain/llava-v1.5-7b')
    # parser.add_argument("--model-path", type=str, default="/home/ma-user/work/llava_ckpt/pretrain/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default="/home/data1/zhangzr22/LLaVA_DATA/sft_forgery_data/NEW_COVER/tampered/57t.jpg")
    parser.add_argument("--mask-image-file", type=str, default="/home/data1/zhangzr22/LLaVA_DATA/sft_forgery_data/NEW_COVER/mask/57t.jpg")
    # parser.add_argument("--mask-image-file", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default='llava_v1')
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
