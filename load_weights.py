import torch

model = YourModelClass()  # 用您自己的模型类进行初始化
model.load_state_dict(torch.load("path_to_model_weights.pth", map_location="cpu"))