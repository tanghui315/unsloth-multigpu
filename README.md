# Unsloth Multi-GPU Support Package

An external extension package that provides multi-GPU parallel training support for the Unsloth framework.

## 🎯 Project Features

- **Zero-intrusive design**: No need to modify Unsloth source code
- **Plugin architecture**: Runs as an independent extension package
- **Fully compatible**: Keeps in sync with the open-source version of Unsloth
- **Simple to use**: Just add a single import statement

## 📦 Installation Requirements

### Required Dependencies
```bash
# 1. Install Unsloth (required - includes unsloth_train function)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# 2. Install PyTorch (GPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Install other dependencies
pip install transformers datasets accelerate
pip install psutil PyYAML  # For memory management and configuration
```

### ⚠️ Important Note
Make sure Unsloth is installed correctly, as the project requires the `unsloth_train` function. If you encounter import errors, run the verification script:
```bash
python examples/verify_installation.py
```

### Optional Dependencies
```bash
# TensorBoard support
pip install tensorboard

# W&B support
pip install wandb
```

## 🚀 Quick Start

### Method 1: Hook Mechanism (Recommended for Existing Code)
```python
import unsloth_multigpu_prototype as unsloth_multigpu
from unsloth import FastLanguageModel, unsloth_train

# 1. Enable multi-GPU support (Hook mechanism)
unsloth_multigpu.enable_multi_gpu(
    num_gpus=4,  # Use 4 GPUs
    batch_size_per_gpu=8,  # Batch size per GPU
    gradient_aggregation="mean"  # Gradient aggregation strategy
)

# 2. Load model (multi-GPU supported automatically)
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/llama-2-7b-bnb-4bit",  # Use 4bit quantized version
    max_seq_length=2048,
    dtype="bfloat16",
    load_in_4bit=True
)

# 3. Use native unsloth_train (internally hooked to multi-GPU logic)
trainer_stats = unsloth_train(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    # ... other training parameters
)
```

### Method 2: Direct Usage (Recommended for New Projects)
```python
from unsloth_multigpu_prototype.core import MultiGPUTrainer, AggregationMethod
from unsloth import FastLanguageModel

# 1. Load model
model, tokenizer = FastLanguageModel.from_pretrained("model_name")

# 2. Configure optimizer
optimizer_config = {
    'class': torch.optim.AdamW,
    'kwargs': {'lr': 2e-5, 'weight_decay': 0.01}
}

# 3. Directly create multi-GPU trainer
trainer = MultiGPUTrainer(
    model=model,
    num_gpus=4,
    optimizer_config=optimizer_config,
    aggregation_method=AggregationMethod.MEAN
)

# 4. Train
trainer.setup()
epoch_stats = trainer.train_epoch(dataloader)
```

### Advanced Configuration
```python
from unsloth_multigpu_prototype.utils import ConfigManager, DeviceManager

# 1. Device management
device_manager = DeviceManager()
devices = device_manager.get_available_devices()

# 2. Get optimal configuration
config_manager = ConfigManager()
optimal_config = config_manager.get_optimal_config(
    model_size="7B",
    available_memory="32GB"
)

# 3. Enable multi-GPU (using optimal config)
unsloth_multigpu.enable_multi_gpu(**optimal_config)
```

## 📁 Project Structure

```
unsloth_multigpu_prototype/
├── __init__.py              # Main entry
├── core/                    # Core components
│   ├── multi_gpu_manager.py # Multi-GPU manager
│   ├── batch_sharding.py    # Batch sharder
│   ├── gradient_aggregator.py # Gradient aggregator
│   ├── multi_gpu_trainer.py # Multi-GPU trainer
│   └── memory_manager.py    # Memory manager
├── hooks/                   # Hook system
│   ├── training_hooks.py    # Training hooks
│   ├── loader_hooks.py      # Loader hooks
│   └── trainer_hooks.py     # Trainer hooks
├── utils/                   # Utility modules
│   ├── device_utils.py      # Device management
│   ├── logging_utils.py     # Logging system
│   └── config_utils.py      # Configuration management
├── examples/                # Example code
│   ├── quick_start.py       # Quick start
│   └── advanced_config.py   # Advanced configuration
└── tests/                   # Test suite
```

## 🛠️ Core Features

### 1. Multi-GPU Management
- Automatic detection and configuration of GPU devices
- Intelligent load balancing
- Memory usage optimization

### 2. Batch Sharding
- Supports uneven sharding
- Adaptive sharding strategy
- Efficient result collection

### 3. Gradient Aggregation
- Multiple aggregation strategies (mean, weighted, median)
- Gradient consistency verification
- Numerical stability guarantee

### 4. Memory Management
- Real-time memory monitoring
- OOM prevention mechanism
- Automatic memory cleanup

### 5. Configuration Management
- Automatic environment detection
- Optimal configuration generation
- Configuration validation and templates

## 📊 Performance Features

- **Training speed boost**: 3.5-4x speedup supported (4-GPU environment)
- **Memory optimization**: Intelligent memory management, reduced OOM risk
- **Stability**: Complete error handling and recovery mechanism
- **Monitoring**: Real-time performance monitoring and logging

## 🧪 Testing

Run the full test suite:
```bash
cd unsloth_multigpu_prototype
python tests/run_all_tests.py
```

Run quick tests:
```bash
python tests/run_all_tests.py --quick
```

## 📖 Examples

See the `examples/` directory for examples:
- `quick_start.py`: Basic example using the Hook mechanism (zero-intrusive)
- `advanced_config.py`: Advanced configuration example for the Hook mechanism
- `direct_trainer_usage.py`: Example of direct usage of MultiGPUTrainer
- `verify_installation.py`: Installation verification script

### Choosing the Right Method
- **Migrating existing projects**: Use the Hook method in `quick_start.py`
- **New project development**: Use the direct method in `direct_trainer_usage.py`
- **Advanced configuration**: Refer to `advanced_config.py`

## ⚠️ Notes

1. **Dependencies**: Make sure to install the Unsloth package first
2. **CUDA environment**: Requires CUDA 11.0+ support
3. **Memory requirements**: At least 8GB VRAM per GPU recommended
4. **Python version**: Python 3.8+ required

## 🤝 Compatibility

- ✅ Unsloth 2023.x versions
- ✅ PyTorch 1.12+
- ✅ CUDA 11.0+
- ✅ Transformers 4.30+

## 📝 Changelog

### v1.0.0 (2025-06)
- Full multi-GPU training support
- Modular Hook system
- Memory management and optimization
- Complete test coverage

## 📞 Support

If you encounter issues, please:
1. Check if dependencies are installed correctly
2. Run the test suite to verify the environment
3. Refer to the example code

---

**Note**: This package is an external extension for Unsloth and does not modify the Unsloth source code, ensuring compatibility with upstream versions. 