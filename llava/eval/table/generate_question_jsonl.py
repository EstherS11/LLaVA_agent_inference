import os
import json
import random
dataset = 'NIST' # Casiav1 NEW_COVER Columbia cocoglide NIST
### 给sft下的所有数据图片提取出来
output_file = f"/home/data1/zhangzr22/LLaVA_DATA/0717/LLaVA_agent/llava/eval/table/forgery_question/question_forgery_{dataset}.jsonl"
root_dir = f'/home/data1/zhangzr22/LLaVA_DATA/sft_forgery_data/{dataset}'
# description_questions = [
#     "Can you describe what is depicted in this image?", "What does this image show?", 
#     "Could you provide a detailed description of this image?", "What is visible in this picture?", 
#     "Please detail the contents of this photograph.", "Describe the scene shown in this image, please.", 
#     "Could you provide a detailed description of this image?", "Can you elaborate on what this image contains?", 
#     "Please provide an overview of what this image depicts.", "Could you provide a detailed description of this image?"
# ]
with open(output_file, 'w') as f:
    for dataset_name in os.listdir(root_dir):
        dataset_path = os.path.join(root_dir, dataset_name)
        if os.path.isdir(dataset_path):
            for subdir, dirs, image_files in os.walk(dataset_path):
                if image_files and 'mask' not in subdir and 'aigc' not in subdir:
                    print(f"Processing directory: {subdir}")
                    for image_file in image_files:
                        question_id = os.path.splitext(image_file)[0]
                        image_path = os.path.join(subdir, image_file)
                        text = 'Is there any forgery in the image?'

                        data = {
                            "question_id": image_path,
                            "image": image_path,
                            "text": text
                        }
                        # breakpoint()
                        json_line = json.dumps(data)
                        f.write(json_line + '\n')