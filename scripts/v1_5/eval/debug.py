import torch



# 加载权重文件
weight_file = '/cache/LLaVA/checkpoints/llava-v1.5-7b-task-lora/adapter_model.bin'
state_dict = torch.load(weight_file)

print(state_dict)
