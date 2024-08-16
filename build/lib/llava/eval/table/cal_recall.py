import json

def calculate_metrics(jsonl_file):
    # 计数器
    TP = 0  # True Positives
    FN = 0  # False Negatives
    FP = 0  # False Positives
    TN = 0  # True Negatives

    with open(jsonl_file, 'r') as file:
        for i, line in enumerate(file):
            data = json.loads(line)
            prediction = data['text']  # 根据实际的键值进行修改

            if i < 203:  # 前203个为有异常
                if "yes" in prediction.lower():
                    TP += 1
                else:
                    FN += 1
            else:  # 之后的为无异常
                if "yes" in prediction.lower():
                    FP += 1
                else:
                    TN += 1

    # 计算召回率和误报率
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    false_positive_rate = FP / (FP + TN) if (FP + TN) != 0 else 0

    return recall, false_positive_rate

# 指定文件路径
file_path = '/cache/LLaVA/llava/eval/table/answer_aoi.jsonl'
recall, fpr = calculate_metrics(file_path)
print(f"召回率 (Recall): {recall:.2f}")
print(f"误报率 (False Positive Rate): {fpr:.2f}")