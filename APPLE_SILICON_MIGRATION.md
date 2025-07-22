# Apple Silicon (M4 Pro) Migration Guide

This document outlines all the changes made to enable HRM (Hierarchical Reasoning Model) to run on Apple Silicon processors, specifically the M4 Pro.

## Summary of Changes

### 1. **Device Abstraction Layer**
- **New file**: `utils/device_utils.py`
- **Purpose**: Provides device-agnostic utilities for CUDA, MPS, and CPU
- **Key functions**:
  - `get_optimal_device()`: Auto-detects best available device
  - `get_device_name()`: Human-readable device names
  - `move_to_device()`: Device-agnostic tensor movement
  - `get_recommended_batch_size()`: Platform-specific batch size optimization

### 2. **FlashAttention Fallback**
- **Modified file**: `models/layers.py`
- **Changes**:
  - Added graceful fallback when FlashAttention is not available
  - Implemented `_standard_attention()` method using PyTorch's native attention
  - Added `FLASH_ATTN_AVAILABLE` flag for runtime detection

### 3. **Training Script Updates**
- **Modified files**: `pretrain.py`, `evaluate.py`
- **Changes**:
  - Replaced hardcoded `cuda()` calls with device-agnostic code
  - Updated distributed training initialization for non-CUDA devices
  - Added automatic batch size adjustment for Apple Silicon
  - Improved device detection and error handling

### 4. **Requirements Updates**
- **Modified file**: `requirements.txt`
- **Changes**:
  - Updated PyTorch installation for Apple Silicon compatibility
  - Added build tools for potential native extensions
  - Maintained backward compatibility with CUDA systems

### 5. **Configuration Files**
- **New files**:
  - `config/cfg_pretrain_apple_silicon.yaml`: Optimized training config
  - `config/arch/hrm_v1_apple_silicon.yaml`: Smaller model architecture
- **Purpose**: Provide Apple Silicon-optimized defaults

### 6. **Setup and Testing**
- **New files**:
  - `setup_apple_silicon.sh`: Automated setup script
  - `test_apple_silicon.py`: Comprehensive compatibility test
- **Purpose**: Streamline setup and verify functionality

### 7. **Documentation Updates**
- **Modified file**: `README.md`
- **Changes**:
  - Added Apple Silicon setup instructions
  - Included platform-specific training commands
  - Added performance expectations and troubleshooting tips

## Key Technical Changes

### Device Management
```python
# Before (CUDA-only)
batch = {k: v.cuda() for k, v in batch.items()}
with torch.device("cuda"):
    model = create_model()

# After (Device-agnostic)
device = get_optimal_device()
batch = move_to_device(batch, device)
with torch.device(device):
    model = create_model()
```

### Attention Mechanism
```python
# Before (FlashAttention only)
attn_output = flash_attn_func(q=query, k=key, v=value, causal=self.causal)

# After (With fallback)
if FLASH_ATTN_AVAILABLE and flash_attn_func is not None:
    attn_output = flash_attn_func(q=query, k=key, v=value, causal=self.causal)
else:
    attn_output = self._standard_attention(query, key, value, batch_size, seq_len)
```

### Distributed Training
```python
# Before (NCCL only)
dist.init_process_group(backend="nccl")
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

# After (Backend selection)
backend = "nccl" if device.type == "cuda" else "gloo"
dist.init_process_group(backend=backend)
if device.type == "cuda":
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
```

## Performance Optimizations for Apple Silicon

### Memory Management
- Unified memory architecture considerations
- Automatic batch size reduction (768 → 96 for training)
- Memory pressure monitoring recommendations

### Model Architecture
- Reduced model size (512 → 384 hidden dimensions)
- Fewer layers (4 → 3 per hierarchy level)
- Adjusted attention heads (8 → 6)

### Compilation
- Disabled `torch.compile` by default for stability
- Environment variable override: `DISABLE_COMPILE=true`

## Usage Instructions

### Quick Setup
```bash
./setup_apple_silicon.sh
```

### Testing
```bash
python test_apple_silicon.py
```

### Training (Sudoku Example)
```bash
# Download dataset
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000

# Train on Apple Silicon
DISABLE_COMPILE=true python pretrain.py \
    data_path=data/sudoku-extreme-1k-aug-1000 \
    epochs=20000 \
    eval_interval=2000 \
    global_batch_size=32 \
    lr=7e-5 \
    puzzle_emb_lr=7e-5 \
    weight_decay=1.0 \
    puzzle_emb_weight_decay=1.0
```

### Training (ARC with optimized config)
```bash
python pretrain.py --config-name=cfg_pretrain_apple_silicon arch=hrm_v1_apple_silicon
```

## Expected Performance

### M4 Pro Performance Estimates
- **Sudoku training**: ~15-20 hours (vs ~10 hours on RTX 4070)
- **Memory usage**: ~8-12GB unified memory
- **Thermal considerations**: May throttle during long runs

### Compatibility Matrix
| Feature | CUDA | MPS (Apple Silicon) | CPU |
|---------|------|-------------------|-----|
| Training | ✅ | ✅ | ✅ |
| FlashAttention | ✅ | ❌ (fallback) | ❌ (fallback) |
| Distributed | ✅ | ❌ | ✅ (limited) |
| torch.compile | ✅ | ⚠️ (disabled) | ✅ |

## Troubleshooting

### Common Issues
1. **MPS fallback warnings**: Normal, operations fall back to CPU when needed
2. **Memory pressure**: Reduce batch size or use smaller model config
3. **Thermal throttling**: Use cooling, reduce workload, or train in segments
4. **Compilation errors**: Ensure `DISABLE_COMPILE=true` is set

### Environment Variables
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1  # Enable MPS fallback
export DISABLE_COMPILE=true           # Disable torch.compile
export OMP_NUM_THREADS=8              # Set CPU thread count
```

## Backward Compatibility

All changes maintain full backward compatibility with CUDA systems. The device detection automatically selects the optimal backend, so existing CUDA workflows continue to work unchanged.
