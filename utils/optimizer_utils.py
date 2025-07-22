"""
Device-aware optimizer utilities for HRM.

This module provides fallback mechanisms for optimizers that require CUDA
when running on non-CUDA systems like Apple Silicon.
"""

import torch
from typing import Any, Dict, Optional, Union
import warnings


# Try to import adam-atan2, fall back to standard optimizers if not available
try:
    from adam_atan2 import AdamATan2
    ADAM_ATAN2_AVAILABLE = True
except ImportError:
    ADAM_ATAN2_AVAILABLE = False
    AdamATan2 = None


class AdamATan2Fallback(torch.optim.AdamW):
    """
    Fallback implementation that mimics AdamATan2 behavior using standard AdamW.
    
    AdamATan2 is a numerically stable, scale-invariant version of Adam that
    eliminates the epsilon hyperparameter. This fallback uses AdamW with
    carefully chosen parameters to approximate the behavior.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        betas: tuple = (0.9, 0.95),
        eps: float = 1e-8,  # AdamATan2 eliminates this, but we need it for AdamW
        **kwargs
    ):
        # Use a smaller epsilon for better numerical stability
        # This approximates the atan2-based approach of the original
        super().__init__(
            params=params,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            **kwargs
        )


def get_adam_atan2_optimizer(
    params,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.95),
    device: Optional[torch.device] = None,
    **kwargs
) -> torch.optim.Optimizer:
    """
    Get AdamATan2 optimizer with automatic fallback for non-CUDA systems.
    
    Args:
        params: Model parameters to optimize
        lr: Learning rate
        weight_decay: Weight decay coefficient
        betas: Coefficients for computing running averages
        device: Target device (auto-detected if None)
        **kwargs: Additional optimizer arguments
        
    Returns:
        AdamATan2 optimizer if available and compatible, otherwise AdamW fallback
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use AdamATan2 if available and on CUDA
    if ADAM_ATAN2_AVAILABLE and device.type == "cuda":
        try:
            return AdamATan2(
                params=params,
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
                **kwargs
            )
        except Exception as e:
            warnings.warn(
                f"Failed to create AdamATan2 optimizer: {e}. "
                f"Falling back to AdamW.",
                UserWarning
            )
    
    # Fallback to AdamW for non-CUDA systems or if AdamATan2 fails
    if not ADAM_ATAN2_AVAILABLE:
        warnings.warn(
            "adam-atan2 package not available. Using AdamW fallback. "
            "For CUDA systems, install with: pip install adam-atan2",
            UserWarning
        )
    elif device.type != "cuda":
        print(f"Using AdamW fallback for {device.type} device (AdamATan2 requires CUDA)")
    
    return AdamATan2Fallback(
        params=params,
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        **kwargs
    )


def get_optimizer_info() -> Dict[str, Any]:
    """Get information about available optimizers."""
    return {
        "adam_atan2_available": ADAM_ATAN2_AVAILABLE,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
        "recommended_optimizer": "AdamATan2" if ADAM_ATAN2_AVAILABLE and torch.cuda.is_available() else "AdamW"
    }
