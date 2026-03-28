import torch
import sys
import os

print("=== PyTorch & CUDA 诊断报告 ===")
print(f"Python版本: {sys.version}")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
print(f"虚拟环境路径: {sys.prefix}")

# 检查CUDA相关路径
cuda_path = os.environ.get('CUDA_PATH')
print(f"CUDA_PATH环境变量: {cuda_path}")

if torch.cuda.is_available():
    print(f"检测到的GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"    显存总量: {props.total_memory / 1024**3:.2f} GB")
else:
    print("\n❌ PyTorch无法检测到CUDA。可能原因:")
    print("1. 安装的是PyTorch CPU版本（最常见）")
    print("2. NVIDIA显卡驱动未安装或版本太旧")
    print("3. CUDA Toolkit未安装或与PyTorch版本不匹配")
    print("4. 虚拟环境中安装的PyTorch与系统CUDA不匹配")