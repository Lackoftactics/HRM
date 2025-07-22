#!/bin/bash

# Setup script for HRM on Apple Silicon (M4 Pro)

echo "Setting up HRM for Apple Silicon (M4 Pro)..."

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "Error: This script is designed for macOS (Apple Silicon)"
    exit 1
fi

# Check if we're on Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo "Warning: This script is optimized for Apple Silicon (arm64), but you're on $(uname -m)"
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch for Apple Silicon
echo "Installing PyTorch for Apple Silicon..."
pip install torch torchvision torchaudio

# Install base requirements (excluding CUDA-dependent packages)
echo "Installing base requirements..."
pip install -r requirements.txt

# Install Apple Silicon specific requirements
echo "Installing Apple Silicon specific requirements..."
pip install -r requirements-apple-silicon.txt

# Try to install adam-atan2, but don't fail if it doesn't work
echo "Attempting to install adam-atan2 (will use fallback if this fails)..."
pip install adam-atan2 || echo "adam-atan2 installation failed - will use AdamW fallback"

# Set environment variables for Apple Silicon optimization
echo "Setting up environment variables..."
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS=$(sysctl -n hw.ncpu)

# Create a simple test script
cat > test_device.py << 'EOF'
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
EOF

# Run the test
echo "Testing device setup..."
python test_device.py

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To run training optimized for Apple Silicon:"
echo "  python pretrain.py --config-name=cfg_pretrain_apple_silicon arch=hrm_v1_apple_silicon"
echo ""
echo "For single GPU demo (Sudoku):"
echo "  DISABLE_COMPILE=true python pretrain.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=20000 eval_interval=2000 global_batch_size=32 lr=7e-5 puzzle_emb_lr=7e-5 weight_decay=1.0 puzzle_emb_weight_decay=1.0"
echo ""
echo "Note: Batch sizes have been automatically adjusted for Apple Silicon memory constraints."
