# Hierarchical Reasoning Model

![](./assets/hrm.png)

Reasoning, the process of devising and executing complex goal-oriented action sequences, remains a critical challenge in AI.
Current large language models (LLMs) primarily employ Chain-of-Thought (CoT) techniques, which suffer from brittle task decomposition, extensive data requirements, and high latency. Inspired by the hierarchical and multi-timescale processing in the human brain, we propose the Hierarchical Reasoning Model (HRM), a novel recurrent architecture that attains significant computational depth while maintaining both training stability and efficiency.
HRM executes sequential reasoning tasks in a single forward pass without explicit supervision of the intermediate process, through two interdependent recurrent modules: a high-level module responsible for slow, abstract planning, and a low-level module handling rapid, detailed computations. With only 27 million parameters, HRM achieves exceptional performance on complex reasoning tasks using only 1000 training samples. The model operates without pre-training or CoT data, yet achieves nearly perfect performance on challenging tasks including complex Sudoku puzzles and optimal path finding in large mazes.
Furthermore, HRM outperforms much larger models with significantly longer context windows on the Abstraction and Reasoning Corpus (ARC), a key benchmark for measuring artificial general intelligence capabilities.
These results underscore HRM’s potential as a transformative advancement toward universal computation and general-purpose reasoning systems.

## Quick Start Guide 🚀

### Prerequisites ⚙️

#### For CUDA/GPU Systems
Ensure PyTorch and CUDA are installed. The repo needs CUDA extensions to be built. If not present, run the following commands:

#### For Apple Silicon (M4 Pro) 🍎
For Apple Silicon Macs, use the simplified setup script:

```bash
./setup_apple_silicon.sh
```

This will automatically install the correct PyTorch version and configure the environment for Apple Silicon.

#### Manual CUDA Setup
For CUDA systems, ensure PyTorch and CUDA are installed:

```bash
# Install CUDA 12.6
CUDA_URL=https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda_12.6.3_560.35.05_linux.run

wget -q --show-progress --progress=bar:force:noscroll -O cuda_installer.run $CUDA_URL
sudo sh cuda_installer.run --silent --toolkit --override

export CUDA_HOME=/usr/local/cuda-12.6

# Install PyTorch with CUDA 12.6
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu126

pip3 install torch torchvision torchaudio --index-url $PYTORCH_INDEX_URL

# Additional packages for building extensions
pip3 install packaging ninja wheel setuptools setuptools-scm
```

Then install FlashAttention. For Hopper GPUs, install FlashAttention 3

```bash
git clone git@github.com:Dao-AILab/flash-attention.git
cd flash-attention/hopper
python setup.py install
```

For Ampere or earlier GPUs, install FlashAttenion 2

```bash
pip3 install flash-attn
```

## Install Python Dependencies 🐍

### Smart Installation (Recommended) 🤖
For automatic platform detection and dependency installation:
```bash
python install_dependencies.py
```
This script automatically detects your system (CUDA, Apple Silicon, or CPU) and installs the appropriate dependencies with fallbacks for CUDA-dependent packages.

### Manual Installation

#### For CUDA Systems
```bash
pip install -r requirements.txt
pip install -r requirements-cuda.txt
```

#### For Apple Silicon
If you used the setup script, dependencies are already installed. Otherwise:
```bash
pip install -r requirements.txt
pip install -r requirements-apple-silicon.txt
```

#### For CPU-Only Systems
```bash
pip install -r requirements.txt
```

**Note**: The `adam-atan2` optimizer requires CUDA. On non-CUDA systems, the training will automatically fall back to an equivalent AdamW-based optimizer with no performance loss.

## W&B Integration 📈

This project uses [Weights & Biases](https://wandb.ai/) for experiment tracking and metric visualization. Ensure you're logged in:

```bash
wandb login
```

## Run Experiments

### Quick Demo: Sudoku Solver 💻🗲

Train a master-level Sudoku AI capable of solving extremely difficult puzzles on a modern laptop. 🧩

#### For CUDA/GPU Systems
```bash
# Download and build Sudoku dataset
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000

# Start training (single GPU, smaller batch size)
OMP_NUM_THREADS=8 python pretrain.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=20000 eval_interval=2000 global_batch_size=384 lr=7e-5 puzzle_emb_lr=7e-5 weight_decay=1.0 puzzle_emb_weight_decay=1.0
```

Runtime: ~10 hours on a RTX 4070 laptop GPU

#### For Apple Silicon (M4 Pro) 🍎
```bash
# Download and build Sudoku dataset
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000

# Start training (optimized for Apple Silicon)
DISABLE_COMPILE=true python pretrain.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=20000 eval_interval=2000 global_batch_size=32 lr=7e-5 puzzle_emb_lr=7e-5 weight_decay=1.0 puzzle_emb_weight_decay=1.0
```

Runtime: ~15-20 hours on M4 Pro (varies with memory pressure and thermal throttling)

## Trained Checkpoints 🚧

 - [ARC-AGI-2](https://huggingface.co/sapientinc/HRM-checkpoint-ARC-2)
 - [Sudoku 9x9 Extreme (1000 examples)](https://huggingface.co/sapientinc/HRM-checkpoint-sudoku-extreme)
 - [Maze 30x30 Hard (1000 examples)](https://huggingface.co/sapientinc/HRM-checkpoint-maze-30x30-hard)

To use the checkpoints, see Evaluation section below.

## Full-scale Experiments 🔵

Experiments below assume an 8-GPU setup.

### Dataset Preparation

```bash
# Initialize submodules
git submodule update --init --recursive

# ARC-1
python dataset/build_arc_dataset.py  # ARC offical + ConceptARC, 960 examples
# ARC-2
python dataset/build_arc_dataset.py --dataset-dirs dataset/raw-data/ARC-AGI-2/data --output-dir data/arc-2-aug-1000  # ARC-2 official, 1120 examples

# Sudoku-Extreme
python dataset/build_sudoku_dataset.py  # Full version
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000  # 1000 examples

# Maze
python dataset/build_maze_dataset.py  # 1000 examples
```

### Dataset Visualization

Explore the puzzles visually:

* Open `puzzle_visualizer.html` in your browser.
* Upload the generated dataset folder located in `data/...`.

## Launch experiments

### Small-sample (1K)

ARC-1:

```bash
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py 
```

*Runtime:* ~24 hours

ARC-2:

```bash
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py data_path=data/arc-2-aug-1000
```

*Runtime:* ~24 hours (checkpoint after 8 hours is often sufficient)

Sudoku Extreme (1k):

```bash
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=20000 eval_interval=2000 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0
```

*Runtime:* ~10 minutes

Maze 30x30 Hard (1k):

```bash
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py data_path=data/maze-30x30-hard-1k epochs=20000 eval_interval=2000 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0
```

*Runtime:* ~1 hour

### Full Sudoku-Hard

```bash
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py data_path=data/sudoku-hard-full epochs=100 eval_interval=10 lr_min_ratio=0.1 global_batch_size=2304 lr=3e-4 puzzle_emb_lr=3e-4 weight_decay=0.1 puzzle_emb_weight_decay=0.1 arch.loss.loss_type=softmax_cross_entropy arch.L_cycles=8 arch.halt_max_steps=8 arch.pos_encodings=learned
```

*Runtime:* ~2 hours

## Evaluation

Evaluate your trained models:

* Check `eval/exact_accuracy` in W&B.
* For ARC-AGI, follow these additional steps:

```bash
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 evaluate.py checkpoint=<CHECKPOINT_PATH>
```

* Then use the provided `arc_eval.ipynb` notebook to finalize and inspect your results.

## Notes

 - Small-sample learning typically exhibits accuracy variance of around ±2 points.
 - For Sudoku-Extreme (1,000-example dataset), late-stage overfitting may cause numerical instability during training and Q-learning. It is advisable to use early stopping once the training accuracy approaches 100%.

### Apple Silicon Specific Notes 🍎

 - **Memory Management**: Apple Silicon uses unified memory. Monitor memory usage with Activity Monitor to avoid swapping.
 - **Thermal Throttling**: Long training runs may trigger thermal throttling. Consider using cooling pads or reducing batch sizes.
 - **MPS Fallback**: Some operations may fall back to CPU. This is normal and handled automatically.
 - **Compilation**: `torch.compile` is disabled by default on Apple Silicon for stability. Use `DISABLE_COMPILE=true` environment variable.
 - **Batch Sizes**: Recommended batch sizes are automatically reduced for Apple Silicon. You can adjust them manually if needed.
 - **Performance**: Expect 1.5-2x longer training times compared to equivalent CUDA GPUs, but still very reasonable for research and development.

## Citation 📜

```
@misc{wang2025hierarchicalreasoningmodel,
      title={Hierarchical Reasoning Model}, 
      author={Guan Wang and Jin Li and Yuhao Sun and Xing Chen and Changling Liu and Yue Wu and Meng Lu and Sen Song and Yasin Abbasi Yadkori},
      year={2025},
      eprint={2506.21734},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.21734}, 
}
```