from sklearn.metrics import roc_auc_score, accuracy_score
import json

def calculate_metrics(jsonl_file):
    # 计数器
    TP = 0  # True Positives
    FN = 0  # False Negatives
    FP = 0  # False Positives
    TN = 0  # True Negatives

    true_labels = []
    predictions = []

    with open(jsonl_file, 'r') as file:
        for i, line in enumerate(file):
            data = json.loads(line)
            prediction = data['text']
            true_label = 'tampered' in data['question_id']  # 判断是否为异常样本

            predictions.append(prediction)
            true_labels.append(true_label)

            if true_label:
                if 'yes' in prediction.lower():
                    TP += 1
                else:
                    FN += 1
            else:
                if 'yes' in prediction.lower():
                    FP += 1
                else:
                    TN += 1

    # 计算召回率和误报率
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    fpr = FP / (FP + TN) if (FP + TN) != 0 else 0

    # 计算 AUC 和准确率
    true_labels = [int(label) for label in true_labels]
    predictions = [int('yes' in p.lower()) for p in predictions]

    auc = roc_auc_score(true_labels, predictions)
    acc = accuracy_score(true_labels, predictions)

    return recall, fpr, auc, acc
## Columbia  NEW_COVER
# 指定文件路径
file_path = '/home/data1/zhangzr22/LLaVA_DATA/0717/LLaVA_agent_inference/llava/eval/table/forgery_question/answer_forgery_NEW_COVER.jsonl'
recall, fpr, auc, acc = calculate_metrics(file_path)
print(f"召回率 (Recall): {recall:.3f}")
print(f"误报率 (False Positive Rate): {fpr:.3f}")
print(f"AUC: {auc:.3f}")
print(f"准确率 (Accuracy): {acc:.3f}")