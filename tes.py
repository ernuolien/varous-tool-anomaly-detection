import torch
import os

# 获取 PyTorch 安装路径
torch_path = os.path.dirname(torch.__file__)
print(f"PyTorch 安装路径: {torch_path}")
# 检查环境变量
print(f"PATH: {os.environ.get('PATH')}")
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}")
import torch

print(f"PyTorch 支持的 CUDA 版本: {torch.version.cuda}")
print(f"是否检测到 CUDA: {torch.cuda.is_available()}")