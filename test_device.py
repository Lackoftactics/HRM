import torch
from utils.device_utils import get_optimal_device, get_device_name

print(f"PyTorch version: {torch.__version__}")
print(f"Optimal device: {get_optimal_device()}")
print(f"Device name: {get_device_name()}")

if torch.backends.mps.is_available():
    print("✅ MPS (Metal Performance Shaders) is available")
    # Test basic tensor operations
    x = torch.randn(10, 10).to('mps')
    y = torch.randn(10, 10).to('mps')
    z = torch.matmul(x, y)
    print("✅ Basic MPS tensor operations work")
else:
    print("❌ MPS is not available, will use CPU")

print("\nSetup complete! You can now run training with:")
print("python pretrain.py --config-name=cfg_pretrain_apple_silicon arch=hrm_v1_apple_silicon")
