#!/usr/bin/env python3
"""
Test script to verify Apple Silicon compatibility for HRM
"""

import torch
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.device_utils import (
    get_optimal_device, get_device_name, setup_device_environment,
    get_recommended_batch_size, should_use_flash_attention,
    get_memory_info
)

def test_device_setup():
    """Test basic device setup"""
    print("🔧 Testing device setup...")
    
    setup_device_environment()
    device = get_optimal_device()
    
    print(f"✅ Optimal device: {device}")
    print(f"✅ Device name: {get_device_name()}")
    print(f"✅ FlashAttention available: {should_use_flash_attention()}")
    
    # Test memory info
    memory_info = get_memory_info()
    print(f"✅ Memory info: {memory_info}")
    
    return device

def test_tensor_operations(device):
    """Test basic tensor operations on the device"""
    print(f"\n🧮 Testing tensor operations on {device}...")
    
    try:
        # Create test tensors
        x = torch.randn(100, 100, device=device)
        y = torch.randn(100, 100, device=device)
        
        # Test basic operations
        z = torch.matmul(x, y)
        print("✅ Matrix multiplication works")
        
        # Test gradients
        x.requires_grad_(True)
        loss = (z ** 2).sum()
        loss.backward()
        print("✅ Gradient computation works")
        
        # Test attention-like operations
        q = torch.randn(2, 10, 8, 64, device=device)
        k = torch.randn(2, 10, 8, 64, device=device)
        v = torch.randn(2, 10, 8, 64, device=device)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (64 ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        print("✅ Attention operations work")
        
        return True
        
    except Exception as e:
        print(f"❌ Tensor operations failed: {e}")
        return False

def test_model_components():
    """Test key model components"""
    print(f"\n🏗️ Testing model components...")
    
    try:
        from models.layers import Attention, SwiGLU, rms_norm
        
        device = get_optimal_device()
        
        # Test RMS norm
        x = torch.randn(2, 10, 512, device=device)
        normed = rms_norm(x, 1e-5)
        print("✅ RMS normalization works")
        
        # Test SwiGLU
        swiglu = SwiGLU(512, 4.0).to(device)
        out = swiglu(x)
        print("✅ SwiGLU activation works")
        
        # Test Attention (this will use fallback on non-CUDA)
        attention = Attention(
            hidden_size=512,
            head_dim=64,
            num_heads=8,
            num_key_value_heads=8,
            causal=False
        ).to(device)
        
        attn_out = attention(x)
        print("✅ Attention layer works")
        
        return True
        
    except Exception as e:
        print(f"❌ Model components test failed: {e}")
        return False

def test_batch_size_recommendations():
    """Test batch size recommendations"""
    print(f"\n📊 Testing batch size recommendations...")
    
    test_sizes = [768, 384, 96, 32]
    
    for size in test_sizes:
        recommended = get_recommended_batch_size(size)
        print(f"✅ Batch size {size} → {recommended}")
    
    return True

def main():
    """Run all tests"""
    print("🍎 HRM Apple Silicon Compatibility Test")
    print("=" * 50)
    
    # Test device setup
    device = test_device_setup()
    
    # Test tensor operations
    tensor_ok = test_tensor_operations(device)
    
    # Test model components
    model_ok = test_model_components()
    
    # Test batch size recommendations
    batch_ok = test_batch_size_recommendations()
    
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"✅ Device setup: OK")
    print(f"{'✅' if tensor_ok else '❌'} Tensor operations: {'OK' if tensor_ok else 'FAILED'}")
    print(f"{'✅' if model_ok else '❌'} Model components: {'OK' if model_ok else 'FAILED'}")
    print(f"{'✅' if batch_ok else '❌'} Batch size recommendations: {'OK' if batch_ok else 'FAILED'}")
    
    if all([tensor_ok, model_ok, batch_ok]):
        print("\n🎉 All tests passed! Your system is ready for HRM training.")
        print("\nNext steps:")
        print("1. Download a dataset: python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000")
        print("2. Start training: DISABLE_COMPILE=true python pretrain.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=20000 eval_interval=2000 global_batch_size=32")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
