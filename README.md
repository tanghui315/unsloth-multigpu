# Unsloth Multi-GPU Support Package

An external extension package that provides multi-GPU parallel training support for the Unsloth framework.

**[中文文档 / Chinese Documentation](README_CN.md)**

## 🎯 Project Features

- **Zero-intrusive design**: No need to modify Unsloth source code
- **Plugin architecture**: Runs as an independent extension package
- **Fully compatible**: Keeps in sync with the open-source version of Unsloth
- **Simple to use**: Just add a single import statement

## 📦 Installation Requirements

### Quick Installation
```bash
# Clone and install the project
git clone https://github.com/tanghui315/unsloth-multigpu.git
cd unsloth-multigpu
pip install .
```

### Dependency Details
This project depends on the following packages (automatically handled during installation):
- Unsloth (provides FastLanguageModel and LoRA support)
- TRL (provides SFTTrainer for supervised fine-tuning)
- PyTorch (GPU version)
- Transformers, datasets, accelerate
- psutil, PyYAML (for memory management and configuration)

### ⚠️ Important Note
Ensure your system has CUDA support. If you encounter import errors, run the verification script:
```bash
python examples/verify_installation.py
```

### Verify Installation
```bash
# Check if package is installed
pip list | grep unsloth-multigpu

# Test import
python -c "import unsloth_multigpu; print('✅ Installation successful!')"
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

#### Running Example
```bash
# Method 1: Single process run (Hook will prompt to use torchrun)
python examples/quick_start.py

# Method 2: Use torchrun for real DDP training (Recommended)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 examples/quick_start.py
```

#### Code Example
```python
import unsloth_multigpu as ump
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# 1. Enable multi-GPU support (Hook mechanism)
ump.enable_multi_gpu(
    num_gpus=2,  # Use 2 GPUs
    batch_size_per_gpu=2,  # Batch size per GPU
    ddp_backend="nccl",  # DDP communication backend
    enable_memory_optimization=True  # Enable memory optimization
)

# 2. Load model (multi-GPU supported automatically)
model, tokenizer = FastLanguageModel.from_pretrained(
    "/path/to/your/model",  # Model path
    max_seq_length=4096,
    dtype=torch.bfloat16,  # Note: use torch.bfloat16 instead of string
    load_in_4bit=True
)

# 3. Configure LoRA (required for unsloth)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# 4. Configure training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,  # This will be handled by multi-GPU
    learning_rate=2e-5,
    logging_steps=1,
    save_strategy="steps",
    save_steps=100,
)

# 5. Create SFTTrainer and train (multi-GPU supported automatically)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",  # or use formatting_func
    args=training_args,
    max_seq_length=4096,
)

trainer_stats = trainer.train()
```

### Method 3: GRPO Reinforcement Learning Training (Multi-GPU Support)

GRPO (Generalized Reinforcement Learning from Policy Optimization) is an advanced reinforcement learning training method with multi-GPU parallel training support.

#### Complete GRPO Training Steps

**Step 1: Start vLLM Inference Service** (if using vLLM inference)
```bash
# Start vLLM service for fast inference (run in separate terminal)
CUDA_VISIBLE_DEVICES=0,1 trl vllm-serve \
    --model /path/to/your/model \
    --tensor-parallel-size 2 \
    --port 8000
```

**Step 2: Enable GRPO Support and Multi-GPU Training**
```bash
# Use torchrun to start multi-GPU GRPO training
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 examples/grpo_open_r1_multi_gpu.py
```

#### GRPO Training Code Example
```python
import unsloth_multigpu as ump
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig

# 1. Enable GRPO support
ump.enable_grpo_support()

# 2. Enable multi-GPU support
ump.enable_multi_gpu(
    num_gpus=2,
    batch_size_per_gpu=1,
    ddp_backend="nccl",
    enable_memory_optimization=True
)

# 3. Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    max_seq_length=8192,
    load_in_4bit=True,
)

# 4. Configure LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=32,
    use_gradient_checkpointing="unsloth",
)

# 5. Configure GRPO training arguments
training_args = GRPOConfig(
    output_dir="./grpo_output",
    num_generations=8,
    learning_rate=5e-6,
    per_device_train_batch_size=1,
    gradient_checkpointing=True,
    use_vllm=True,  # Use vLLM for accelerated inference
    vllm_server_host="127.0.0.1",
    vllm_server_port=8000,
)

# 6. Prepare reward functions (requires open-r1 support)
from open_r1.rewards import get_reward_funcs
reward_funcs = get_reward_funcs(script_args)

# 7. Create GRPO trainer
trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_funcs,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

# 8. Start training
trainer.train()
```

#### GRPO Training Configuration Example (YAML)
```yaml
# examples/configs/grpo_open_r1_config.yaml
model_name_or_path: Qwen/Qwen2.5-32B-Instruct
dataset_name: data/grpo_data.jsonl

# GRPO specific configuration
use_vllm: true
vllm_server_host: 127.0.0.1
vllm_server_port: 8000
num_generations: 8
learning_rate: 5.0e-06
per_device_train_batch_size: 1

# Reward function configuration
reward_funcs:
  - accuracy
  - format
  - reasoning_steps
  - cosine
  - repetition_penalty
  - length

# LoRA configuration
lora_r: 16
gradient_checkpointing: true
```

#### Important: Difference Between Two Running Methods

1. **Single Process Run** (`python script.py`):
   - Hook detects multi-GPU requirement and prompts to use torchrun
   - Shows error message and correct launch command
   - Suitable for testing if configuration is correct

2. **Torchrun Execution** (`torchrun --nproc_per_node=2 script.py`):
   - Launches real DDP multi-process training
   - Each GPU runs an independent process
   - Automatic data splitting and gradient synchronization
   - Achieves real performance improvement

```bash
# If you run the regular command, you'll see a prompt like:
$ python examples/quick_start.py
❌ Multi-GPU training requires launching with torchrun
💡 Please use the following command to launch:
   CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 examples/quick_start.py
```

### Method 2: Direct Usage (Recommended for New Projects)
```python
from unsloth_multigpu.core import MultiGPUTrainer, AggregationMethod
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
from unsloth_multigpu.utils import ConfigManager, DeviceManager

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
ump.enable_multi_gpu(**optimal_config)
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

### 3. DDP Training Support
- PyTorch native DDP implementation
- Automatic gradient synchronization
- Efficient NCCL communication

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
cd unsloth-multigpu
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
- `grpo_open_r1_multi_gpu.py`: GRPO reinforcement learning multi-GPU training example
- `configs/grpo_open_r1_config.yaml`: GRPO training YAML configuration example
- `verify_installation.py`: Installation verification script

### Choosing the Right Method
- **Migrating existing projects**: Use the Hook method in `quick_start.py`
- **New project development**: Use the direct method in `direct_trainer_usage.py`
- **GRPO reinforcement learning training**: Use `grpo_open_r1_multi_gpu.py` for multi-GPU GRPO training
- **Advanced configuration**: Refer to `advanced_config.py`

## ⚠️ Notes

1. **Dependencies**: Make sure to install the Unsloth package first
2. **CUDA environment**: Requires CUDA 11.0+ support
3. **Memory requirements**: At least 8GB VRAM per GPU recommended
4. **Python version**: Python 3.8+ required
5. **DDP Training**: Multi-GPU training requires launching with torchrun
6. **Memory Optimization**: Recommend using `load_in_4bit=True` to reduce VRAM usage
7. **GRPO Training**: When using vLLM, start the inference service first, supports open-r1 professional reward functions
8. **vLLM Service**: Start vLLM service for fast inference before GRPO training

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

## 🌐 Language Support

- **English**: This document
- **中文**: [README_CN.md](README_CN.md)

## 📞 Support & Contributing

- **GitHub Repository**: [https://github.com/tanghui315/unsloth-multigpu](https://github.com/tanghui315/unsloth-multigpu)
- **Issues**: [Report bugs or request features](https://github.com/tanghui315/unsloth-multigpu/issues)
- **Discussions**: [Community discussions](https://github.com/tanghui315/unsloth-multigpu/discussions)

If you encounter issues, please:
1. Check if dependencies are installed correctly
2. Run the test suite to verify the environment
3. Refer to the example code
4. Search existing issues or create a new one

---

**Note**: This package is an external extension for Unsloth and does not modify the Unsloth source code, ensuring compatibility with upstream versions. 