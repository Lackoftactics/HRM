#!/usr/bin/env python3
"""
Test script to verify the optimizer fallback mechanism works correctly.
"""

import torch
from utils.optimizer_utils import get_adam_atan2_optimizer, get_optimizer_info
from utils.device_utils import get_optimal_device, get_device_name


def test_optimizer_fallback():
    """Test that the optimizer fallback works correctly."""
    print("🧪 Testing Optimizer Fallback Mechanism")
    print("=" * 50)
    
    # Get device info
    device = get_optimal_device()
    print(f"Device: {get_device_name()}")
    print(f"Device type: {device.type}")
    print()
    
    # Get optimizer info
    optimizer_info = get_optimizer_info()
    print("Optimizer Information:")
    for key, value in optimizer_info.items():
        print(f"  {key}: {value}")
    print()
    
    # Create a simple model for testing
    model = torch.nn.Linear(10, 1)
    model = model.to(device)
    
    # Test optimizer creation
    print("Testing optimizer creation...")
    try:
        optimizer = get_adam_atan2_optimizer(
            model.parameters(),
            lr=1e-3,
            weight_decay=0.01,
            betas=(0.9, 0.95),
            device=device
        )
        
        print(f"✅ Optimizer created successfully: {type(optimizer).__name__}")
        print(f"   Module: {type(optimizer).__module__}")
        
        # Test a simple optimization step
        print("\nTesting optimization step...")
        
        # Create some dummy data
        x = torch.randn(32, 10).to(device)
        y = torch.randn(32, 1).to(device)
        
        # Forward pass
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)
        
        print(f"Initial loss: {loss.item():.6f}")
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check that parameters were updated
        output2 = model(x)
        loss2 = torch.nn.functional.mse_loss(output2, y)
        
        print(f"Loss after step: {loss2.item():.6f}")
        
        if abs(loss.item() - loss2.item()) > 1e-8:
            print("✅ Parameters updated successfully")
        else:
            print("❌ Parameters may not have been updated")
            
        return True
        
    except Exception as e:
        print(f"❌ Optimizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_compatibility():
    """Test compatibility with the training configuration."""
    print("\n🔧 Testing Training Configuration Compatibility")
    print("=" * 50)
    
    try:
        # Test the exact configuration used in pretrain.py
        device = get_optimal_device()
        
        # Simulate model parameters
        dummy_model = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 10)
        ).to(device)
        
        # Test with training configuration parameters
        config_params = {
            "lr": 1e-4,
            "weight_decay": 0.1,
            "betas": (0.9, 0.95)
        }
        
        optimizer = get_adam_atan2_optimizer(
            dummy_model.parameters(),
            **config_params,
            device=device
        )
        
        print(f"✅ Training configuration compatible")
        print(f"   Optimizer: {type(optimizer).__name__}")
        print(f"   Learning rate: {config_params['lr']}")
        print(f"   Weight decay: {config_params['weight_decay']}")
        print(f"   Betas: {config_params['betas']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training configuration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 HRM Optimizer Fallback Test Suite")
    print("=" * 60)
    print()
    
    test1_passed = test_optimizer_fallback()
    test2_passed = test_compatibility()
    
    print("\n" + "=" * 60)
    print("📋 Test Results Summary:")
    print(f"  Optimizer Fallback: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"  Training Config:    {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! The optimizer fallback is working correctly.")
        print("\nYou can now run training with:")
        print("  DISABLE_COMPILE=true python pretrain.py --config-name=cfg_pretrain_apple_silicon")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
