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

        # Test gradients (with proper tensor creation for gradients)
        x_grad = torch.randn(100, 100, device=device, requires_grad=True)
        y_grad = torch.randn(100, 100, device=device)
        z_grad = torch.matmul(x_grad, y_grad)
        loss = (z_grad ** 2).sum()
        loss.backward()
        print("✅ Gradient computation works")

        # Test attention-like operations with proper dtypes
        batch_size, seq_len, num_heads, head_dim = 2, 10, 8, 64
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float32)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float32)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float32)

        # Compute attention scores
        scale = 1.0 / (head_dim ** 0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply softmax
        attn = torch.softmax(scores, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn, v)
        print("✅ Attention operations work")

        # Test mixed precision operations if supported
        if device.type in ["cuda", "mps"]:
            try:
                x_half = torch.randn(50, 50, device=device, dtype=torch.float16)
                y_half = torch.randn(50, 50, device=device, dtype=torch.float16)
                z_half = torch.matmul(x_half, y_half)
                print("✅ Half precision operations work")
            except Exception as e:
                print(f"⚠️  Half precision operations not fully supported: {e}")

        return True

    except Exception as e:
        print(f"❌ Tensor operations failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_components():
    """Test key model components"""
    print(f"\n🏗️ Testing model components...")

    try:
        from models.layers import Attention, SwiGLU, rms_norm, RotaryEmbedding

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

        # Create proper cos_sin for RoPE (or None for no positional encoding)
        # For testing, we'll use None to test without positional encoding
        cos_sin = None

        # Test with proper signature: forward(cos_sin, hidden_states)
        attn_out = attention(cos_sin, x)
        print("✅ Attention layer works")

        # Test with RoPE positional encoding
        # Create RoPE with matching sequence length (10 to match our test tensor)
        rope = RotaryEmbedding(dim=64, max_position_embeddings=10, base=10000.0, device=device)
        cos_sin = rope()

        # Verify cos_sin shapes match our sequence length
        cos, sin = cos_sin
        if cos.shape[0] >= x.shape[1]:  # seq_len dimension
            attn_out_rope = attention(cos_sin, x)
            print("✅ Attention with RoPE works")
        else:
            print("⚠️  RoPE test skipped due to sequence length mismatch (expected behavior)")

        return True

    except Exception as e:
        print(f"❌ Model components test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_size_recommendations():
    """Test batch size recommendations"""
    print(f"\n📊 Testing batch size recommendations...")

    test_sizes = [768, 384, 96, 32]

    for size in test_sizes:
        recommended = get_recommended_batch_size(size)
        print(f"✅ Batch size {size} → {recommended}")

    return True


def test_optimizer_integration():
    """Test optimizer integration with the new fallback mechanism"""
    print(f"\n🔧 Testing optimizer integration...")

    try:
        from utils.optimizer_utils import get_adam_atan2_optimizer, get_optimizer_info

        device = get_optimal_device()

        # Get optimizer info
        optimizer_info = get_optimizer_info()
        print(f"✅ Optimizer info: {optimizer_info['recommended_optimizer']}")

        # Create a simple model for testing
        model = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 10)
        ).to(device)

        # Test optimizer creation with training parameters
        optimizer = get_adam_atan2_optimizer(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            device=device
        )

        print(f"✅ Optimizer created: {type(optimizer).__name__}")

        # Test a simple optimization step
        x = torch.randn(32, 512, device=device)
        y = torch.randn(32, 10, device=device)

        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("✅ Optimizer step completed successfully")

        return True

    except Exception as e:
        print(f"❌ Optimizer integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

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

    # Test optimizer integration
    optimizer_ok = test_optimizer_integration()

    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"✅ Device setup: OK")
    print(f"{'✅' if tensor_ok else '❌'} Tensor operations: {'OK' if tensor_ok else 'FAILED'}")
    print(f"{'✅' if model_ok else '❌'} Model components: {'OK' if model_ok else 'FAILED'}")
    print(f"{'✅' if batch_ok else '❌'} Batch size recommendations: {'OK' if batch_ok else 'FAILED'}")
    print(f"{'✅' if optimizer_ok else '❌'} Optimizer integration: {'OK' if optimizer_ok else 'FAILED'}")

    if all([tensor_ok, model_ok, batch_ok, optimizer_ok]):
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
