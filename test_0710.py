mask_path = '/home/ma-user/work/LLaVA/data/sft_forgery_data/fogery_inpaint/authentic/000000011426.jpg'
from torchvision import transforms
from PIL import Image
mask_transform = transforms.Compose([
        transforms.Resize((336, 336)),
        transforms.CenterCrop(336),
        transforms.ToTensor()
    ])

def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
import numpy as np                    
# mask_image = Image.open(mask_path).convert('RGB')
mask_image = np.array(Image.open(mask_path).convert('L')) > 0
mask_image = Image.fromarray(mask_image.astype(np.uint8) * 255, mode='L')
# mask_image = expand2square(mask_image, tuple(int(x*255) for x in processor.image_mean))
breakpoint()
mask_image_gt = mask_transform(mask_image)