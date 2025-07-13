# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **unsloth-multigpu**, an external extension package that adds multi-GPU parallel training support to the Unsloth framework. It's designed as a zero-intrusive plugin that doesn't modify Unsloth source code, maintaining full compatibility with the open-source version.

## Core Architecture

### DDP Implementation - Based on PyTorch Native DistributedDataParallel
The package provides high-performance multi-GPU training using PyTorch's native DDP:

- **DDPManager** (`unsloth_multigpu/core/ddp_manager.py`): PyTorch DDP process group management
- **DDPTrainer** (`unsloth_multigpu/core/ddp_trainer.py`): True parallel training with automatic gradient synchronization
- **DDPLauncher** (`unsloth_multigpu/core/ddp_launcher.py`): Multi-process training coordinator
- **MemoryManager** (`unsloth_multigpu/core/memory_manager.py`): OOM prevention and memory optimization
- **Performance**: 1.8-3.5x speedup over single GPU (real parallel processing)

### Hook-Based Plugin System
- **Training Hooks** (`unsloth_multigpu/hooks/training_hooks.py`): Intercept training functions, route to DDP
- **Loader Hooks** (`unsloth_multigpu/hooks/loader_hooks.py`): Manage data loading across GPUs  
- **Trainer Hooks** (`unsloth_multigpu/hooks/trainer_hooks.py`): Wrap SFTTrainer functionality

## Development Commands

### Installation and Setup
```bash
# Install package in development mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"

# Install with full features
pip install -e ".[full]"

# Quick installation verification
python -c "import unsloth_multigpu; print('✅ Installation successful!')"
```

### Testing
```bash
# Run all tests
python tests/run_all_tests.py

# Run quick tests only
python tests/run_all_tests.py --quick

# Run specific test file
python -m pytest tests/test_core_components.py

# Run with console script
unsloth-multigpu-test
```

### Code Quality (Development Dependencies)
```bash
# Format code with black
black unsloth_multigpu/ tests/ examples/

# Lint with flake8
flake8 unsloth_multigpu/ tests/ examples/

# Type checking with mypy
mypy unsloth_multigpu/

# Run tests with coverage
pytest --cov=unsloth_multigpu tests/
```

### Examples
```bash
# DDP multi-GPU training
CUDA_VISIBLE_DEVICES=0,1 python examples/ddp_quick_start.py

# Basic multi-GPU training example
CUDA_VISIBLE_DEVICES=0,1 python examples/quick_start.py

# Advanced configuration example
python examples/advanced_config.py

# Verify installation
python examples/verify_installation.py
```

## Key Usage Patterns

### Hook Method (Zero Code Change)
```python
import unsloth_multigpu as ump

# Enable DDP multi-GPU support
ump.enable_multi_gpu(
    num_gpus=2,
    batch_size_per_gpu=2,
    ddp_backend="nccl"  # nccl for GPU, gloo for CPU
)

# Regular Unsloth code works automatically with DDP
model, tokenizer = FastLanguageModel.from_pretrained(...)
trainer = SFTTrainer(...)  # Automatically uses PyTorch DDP
trainer.train()  # 1.8-3.5x speedup expected
```

### Direct DDP Method (For Advanced Users)
```python
from unsloth_multigpu.core import launch_ddp_training
from unsloth_multigpu.utils import MultiGPUConfig

config = MultiGPUConfig(num_gpus=2, batch_size_per_gpu=2)
result = launch_ddp_training(trainer, config)
```

## Configuration Management

The package uses `MultiGPUConfig` class for configuration. Key parameters:
- `num_gpus`: Number of GPUs to use
- `batch_size_per_gpu`: Batch size per GPU device
- `ddp_backend`: DDP communication backend (nccl/gloo)
- `memory_optimization`: Enable memory management features
- `enable_gradient_checkpointing`: Use gradient checkpointing
- `find_unused_parameters`: DDP find_unused_parameters setting

## Dependencies and Requirements

### Core Dependencies
- torch>=1.12.0
- transformers>=4.30.0
- accelerate>=0.20.0
- datasets>=2.0.0
- unsloth (installed separately)
- trl (for SFTTrainer)

### System Requirements
- Python 3.8+
- CUDA 11.0+
- At least 8GB VRAM per GPU recommended
- Multi-GPU environment (2+ GPUs)

## Important Notes

1. **Unsloth Dependency**: Unsloth must be installed separately as it's not included in requirements.txt
2. **CUDA Environment**: Package requires CUDA and will fail gracefully on CPU-only systems
3. **Memory Management**: The package includes OOM prevention mechanisms
4. **Zero-Intrusive Design**: No modification of Unsloth source code required
5. **Hook Lifecycle**: Always call `ump.disable_multi_gpu()` when done to clean up hooks

## Troubleshooting

- Run `python examples/verify_installation.py` for installation issues
- Check CUDA availability with `torch.cuda.is_available()`
- Use debug mode: `ump.enable_multi_gpu(debug=True)`
- Monitor status: `ump.get_multi_gpu_status()`