import os
import json

image_dir = "/cache/zfr/07_AOI/zhuangbei_aoi/data/test"
output_file = "/cache/LLaVA/llava/eval/table/question_aoi.jsonl"

# 获取test目录下的所有图片文件名
image_files = [file for file in os.listdir(image_dir) if file.endswith(".jpg")]

with open(output_file, 'w') as f:
    for image_file in image_files:
        question_id = os.path.splitext(image_file)[0]
        image_path = os.path.join(image_dir, image_file)
        text = "This is a photo of modules for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part.\nIs there any anomaly in this image?"

        data = {
            "question_id": question_id,
            "image": image_path,
            "text": text
        }

        json_line = json.dumps(data)
        f.write(json_line + '\n')


image_dir = "/cache/zfr/07_AOI/zhuangbei_aoi/data/OK"
output_file = "/cache/LLaVA/llava/eval/table/question_aoi_neg.jsonl"

# 获取test目录下的所有图片文件名
image_files = [file for file in os.listdir(image_dir) if file.endswith(".jpg")]

with open(output_file, 'w') as f:
    for image_file in image_files:
        question_id = os.path.splitext(image_file)[0]
        image_path = os.path.join(image_dir, image_file)
        text = "This is a photo of modules for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part.\nIs there any anomaly in this image?"

        data = {
            "question_id": question_id,
            "image": image_path,
            "text": text
        }

        json_line = json.dumps(data)
        f.write(json_line + '\n')