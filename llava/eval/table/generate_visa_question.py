import os
import json
import random
import csv
### 给sft下的所有数据图片提取出来
output_file = "/home/ma-user/work/LLaVA/llava/eval/table/question_anomaly_visa_test.jsonl"

forgery_questions = [
"Is there any anomaly in this image?", "Is there any anomaly in this picture?"
]

CLASS_NAMES = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2','pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
datas_csv_path = '/cache/VisA/split_csv/1cls.csv'

root_dir = '/cache/VisA'
root_img = 'fogery_img_visa'
paths = []

# breakpoint()
with open(datas_csv_path, 'r') as file:
    reader = csv.reader(file)

    for row in reader:
        if row[1] == 'test' and row[0] in CLASS_NAMES:
            file_path = os.path.join(root_dir, row[3])
            paths.append(file_path)

with open(output_file, 'w') as f:

    for image_path in paths:
        question_id = os.path.splitext(image_path)[0]
        text = random.choice(forgery_questions)

        data = {
            "question_id": image_path,
            "image": image_path,
            "text": text
        }
        # breakpoint()
        json_line = json.dumps(data)
        f.write(json_line + '\n')