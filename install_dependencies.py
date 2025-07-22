#!/usr/bin/env python3
"""
Smart dependency installer for HRM that handles CUDA vs non-CUDA systems.

This script automatically detects the system capabilities and installs
the appropriate dependencies, with fallbacks for Apple Silicon and other
non-CUDA systems.
"""

import subprocess
import sys
import platform
import os
from pathlib import Path


def run_command(cmd, check=True, capture_output=False):
    """Run a shell command with error handling."""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            check=check, 
            capture_output=capture_output,
            text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        if check:
            print(f"Command failed: {e}")
            raise
        return e


def detect_system():
    """Detect system capabilities."""
    system_info = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "is_apple_silicon": platform.system() == "Darwin" and platform.machine() == "arm64",
        "is_linux": platform.system() == "Linux",
        "is_windows": platform.system() == "Windows",
    }

    # Check for CUDA availability (only after torch is installed)
    system_info["cuda_available"] = None  # Will check later
    system_info["mps_available"] = None   # Will check later

    return system_info


def check_torch_capabilities():
    """Check torch capabilities after installation."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        mps_available = (
            hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        )
        return cuda_available, mps_available
    except ImportError:
        return False, False


def install_base_requirements():
    """Install base requirements that work on all platforms."""
    print("Installing base requirements...")
    run_command("pip install -r requirements.txt")


def install_cuda_requirements():
    """Install CUDA-specific requirements."""
    print("Installing CUDA-specific requirements...")
    run_command("pip install -r requirements-cuda.txt")


def install_apple_silicon_requirements():
    """Install Apple Silicon-specific requirements."""
    print("Installing Apple Silicon-specific requirements...")
    run_command("pip install -r requirements-apple-silicon.txt")


def try_install_adam_atan2():
    """Try to install adam-atan2, return True if successful."""
    print("Attempting to install adam-atan2...")
    result = run_command("pip install adam-atan2", check=False)
    if result.returncode == 0:
        print("✅ adam-atan2 installed successfully")
        return True
    else:
        print("❌ adam-atan2 installation failed (will use AdamW fallback)")
        return False


def main():
    """Main installation logic."""
    print("🚀 HRM Dependency Installer")
    print("=" * 50)
    
    # Detect system
    system_info = detect_system()
    print(f"Platform: {system_info['platform']} ({system_info['machine']})")
    print(f"Apple Silicon: {system_info['is_apple_silicon']}")
    print()

    # Install base requirements first
    install_base_requirements()

    # Check torch capabilities after installation
    cuda_available, mps_available = check_torch_capabilities()
    print(f"CUDA Available: {cuda_available}")
    print(f"MPS Available: {mps_available}")
    print()

    # Platform-specific installations
    if system_info["is_apple_silicon"]:
        print("🍎 Detected Apple Silicon - installing optimized dependencies")
        install_apple_silicon_requirements()
        adam_atan2_success = try_install_adam_atan2()

        if not adam_atan2_success:
            print("ℹ️  Using AdamW fallback optimizer (recommended for Apple Silicon)")

    elif cuda_available:
        print("🔥 Detected CUDA - installing CUDA-optimized dependencies")
        install_cuda_requirements()
        adam_atan2_success = try_install_adam_atan2()

        if not adam_atan2_success:
            print("⚠️  adam-atan2 failed to install on CUDA system")
            print("   This may indicate a CUDA setup issue")

    else:
        print("💻 Detected CPU-only system")
        adam_atan2_success = try_install_adam_atan2()

        if not adam_atan2_success:
            print("ℹ️  Using AdamW fallback optimizer (recommended for CPU)")
    
    print()
    print("✅ Installation complete!")
    print()
    print("Next steps:")
    if system_info["is_apple_silicon"]:
        print("  Run: python test_apple_silicon.py")
        print("  Train: DISABLE_COMPILE=true python pretrain.py --config-name=cfg_pretrain_apple_silicon")
    else:
        print("  Run: python -c \"from utils.optimizer_utils import get_optimizer_info; print(get_optimizer_info())\"")
        print("  Train: python pretrain.py")


if __name__ == "__main__":
    main()
