"""
Device utilities for cross-platform compatibility (CUDA, MPS, CPU)
"""
import torch
import os
from typing import Optional


def get_optimal_device() -> torch.device:
    """
    Get the optimal device for the current platform.
    Priority: CUDA > MPS > CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_name() -> str:
    """Get a human-readable device name"""
    device = get_optimal_device()
    if device.type == "cuda":
        return f"CUDA ({torch.cuda.get_device_name()})"
    elif device.type == "mps":
        return "Apple Silicon MPS"
    else:
        return "CPU"


def setup_device_environment():
    """Setup device-specific environment variables and settings"""
    device = get_optimal_device()
    
    if device.type == "mps":
        # MPS-specific optimizations
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        # Reduce memory pressure on Apple Silicon
        torch.mps.empty_cache()
    elif device.type == "cpu":
        # CPU-specific optimizations
        # Use all available cores for CPU training
        if "OMP_NUM_THREADS" not in os.environ:
            import multiprocessing
            os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())


def move_to_device(obj, device: Optional[torch.device] = None):
    """Move tensor or dict of tensors to device"""
    if device is None:
        device = get_optimal_device()
    
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(item, device) for item in obj)
    else:
        return obj


def get_recommended_batch_size(base_batch_size: int) -> int:
    """Get recommended batch size based on device capabilities"""
    device = get_optimal_device()
    
    if device.type == "cuda":
        return base_batch_size
    elif device.type == "mps":
        # Apple Silicon has unified memory, but may need smaller batches
        return max(1, base_batch_size // 2)
    else:
        # CPU training typically needs much smaller batches
        return max(1, base_batch_size // 8)


def supports_distributed_training() -> bool:
    """Check if the current setup supports distributed training"""
    return torch.cuda.is_available() and torch.cuda.device_count() > 1


def get_compile_mode():
    """Get appropriate torch.compile mode for the device"""
    device = get_optimal_device()
    
    if device.type == "cuda":
        return "default"
    elif device.type == "mps":
        # MPS compile support is still experimental
        return None  # Disable compilation for MPS
    else:
        return "default"  # CPU compilation is generally stable


def should_use_flash_attention() -> bool:
    """Check if FlashAttention should be used"""
    # FlashAttention only works with CUDA
    return torch.cuda.is_available()


def get_memory_info():
    """Get memory information for the current device"""
    device = get_optimal_device()
    
    if device.type == "cuda":
        return {
            "total": torch.cuda.get_device_properties(0).total_memory,
            "allocated": torch.cuda.memory_allocated(),
            "cached": torch.cuda.memory_reserved()
        }
    elif device.type == "mps":
        return {
            "total": "Unified Memory",
            "allocated": torch.mps.current_allocated_memory() if hasattr(torch.mps, 'current_allocated_memory') else "N/A",
            "cached": "N/A"
        }
    else:
        import psutil
        return {
            "total": psutil.virtual_memory().total,
            "allocated": "N/A",
            "cached": "N/A"
        }
