# Unsloth多GPU支持库

为Unsloth框架提供多GPU并行训练支持的外部扩展包。

## 🎯 项目特点

- **零侵入性设计**: 无需修改Unsloth源码
- **插件化架构**: 作为独立扩展包运行
- **完全兼容**: 与Unsloth开源版本保持同步
- **简单易用**: 只需添加一行导入语句

## 📦 安装要求

### 快速安装
```bash
# 克隆项目并安装
git clone https://github.com/tanghui315/unsloth-multigpu.git
cd unsloth-multigpu
pip install .
```

### 详细依赖要求
本项目依赖以下包（安装时会自动处理）：
- Unsloth (提供FastLanguageModel和LoRA支持)
- TRL (提供SFTTrainer用于监督式微调)
- PyTorch (GPU版本)
- Transformers, datasets, accelerate
- psutil, PyYAML（用于内存管理和配置）

### ⚠️ 重要说明
确保系统中有CUDA支持。如果遇到导入错误，请运行验证脚本：
```bash
python examples/verify_installation.py
```

### 可选依赖
```bash
# TensorBoard支持
pip install tensorboard

# W&B支持
pip install wandb
```

## 🚀 快速开始

### 方式1: Hook机制（推荐用于现有代码）

#### 运行示例
```bash
# 方式1: 单进程运行（Hook会提示使用torchrun）
python examples/quick_start.py

# 方式2: 使用torchrun进行真正的DDP训练（推荐）
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 examples/quick_start.py
```

#### 代码示例
```python
import unsloth_multigpu as unsloth_multigpu
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# 1. 启用多GPU支持（Hook机制）
unsloth_multigpu.enable_multi_gpu(
    num_gpus=2,  # 使用2个GPU
    batch_size_per_gpu=2,  # 每个GPU的批次大小
    ddp_backend="nccl",  # DDP通信后端
    enable_memory_optimization=True  # 启用内存优化
)

# 2. 加载模型（自动支持多GPU）
model, tokenizer = FastLanguageModel.from_pretrained(
    "/path/to/your/model",  # 模型路径
    max_seq_length=4096,
    dtype=torch.bfloat16,  # 注意：使用torch.bfloat16而不是字符串
    load_in_4bit=True
)

# 3. 配置LoRA（unsloth训练必需）
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA秩
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# 4. 配置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,  # 这将被多GPU处理
    learning_rate=2e-5,
    logging_steps=1,
    save_strategy="steps",
    save_steps=100,
)

# 5. 创建SFTTrainer并训练（自动支持多GPU）
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",  # 或使用formatting_func
    args=training_args,
    max_seq_length=4096,
)

trainer_stats = trainer.train()
```

### 方式3: GRPO强化学习训练（Multi-GPU支持）

GRPO (Generalized Reinforcement Learning from Policy Optimization) 是一种先进的强化学习训练方法，支持多GPU并行训练。

#### GRPO训练完整步骤

**步骤1: 启动vLLM推理服务**（如果使用vLLM推理）
```bash
# 启动vLLM服务用于快速推理（在单独终端中运行）
CUDA_VISIBLE_DEVICES=0,1 trl vllm-serve \
    --model /path/to/your/model \
    --tensor-parallel-size 2 \
    --port 8000
```

**步骤2: 启用GRPO支持并进行多GPU训练**
```bash
# 使用torchrun启动多GPU GRPO训练
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 examples/grpo_open_r1_multi_gpu.py
```

#### GRPO训练代码示例
```python
import unsloth_multigpu as ump
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig

# 1. 启用GRPO支持
ump.enable_grpo_support()

# 2. 启用多GPU支持
ump.enable_multi_gpu(
    num_gpus=2,
    batch_size_per_gpu=1,
    ddp_backend="nccl",
    enable_memory_optimization=True
)

# 3. 加载模型
model, tokenizer = FastLanguageModel.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    max_seq_length=8192,
    load_in_4bit=True,
)

# 4. 配置LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=32,
    use_gradient_checkpointing="unsloth",
)

# 5. 配置GRPO训练参数
training_args = GRPOConfig(
    output_dir="./grpo_output",
    num_generations=8,
    learning_rate=5e-6,
    per_device_train_batch_size=1,
    gradient_checkpointing=True,
    use_vllm=True,  # 使用vLLM加速推理
    vllm_server_host="127.0.0.1",
    vllm_server_port=8000,
)

# 6. 准备奖励函数（需要open-r1支持）
from open_r1.rewards import get_reward_funcs
reward_funcs = get_reward_funcs(script_args)

# 7. 创建GRPO训练器
trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_funcs,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

# 8. 开始训练
trainer.train()
```

#### GRPO训练配置文件示例（YAML）
```yaml
# examples/configs/grpo_open_r1_config.yaml
model_name_or_path: Qwen/Qwen2.5-32B-Instruct
dataset_name: data/grpo_data.jsonl

# GRPO特定配置
use_vllm: true
vllm_server_host: 127.0.0.1
vllm_server_port: 8000
num_generations: 8
learning_rate: 5.0e-06
per_device_train_batch_size: 1

# 奖励函数配置
reward_funcs:
  - accuracy
  - format
  - reasoning_steps
  - cosine
  - repetition_penalty
  - length

# LoRA配置
lora_r: 16
gradient_checkpointing: true
```

#### 重要说明：两种运行方式的区别

1. **单进程运行**（`python script.py`）：
   - Hook检测到多GPU需求，会提示使用torchrun
   - 显示错误信息和正确的启动命令
   - 适合测试配置是否正确

2. **torchrun运行**（`torchrun --nproc_per_node=2 script.py`）：
   - 启动真正的DDP多进程训练
   - 每个GPU运行一个独立进程
   - 自动数据分割和梯度同步
   - 获得真正的性能提升

```bash
# 如果运行普通命令，会看到类似提示：
$ python examples/quick_start.py
❌ 多GPU训练需要使用torchrun启动
💡 请使用以下命令启动:
   CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 examples/quick_start.py
```

### 方式2: 直接使用（推荐用于新项目）
```python
from unsloth_multigpu.core import MultiGPUTrainer, AggregationMethod
from unsloth import FastLanguageModel

# 1. 加载模型
model, tokenizer = FastLanguageModel.from_pretrained("model_name")

# 2. 配置优化器
optimizer_config = {
    'class': torch.optim.AdamW,
    'kwargs': {'lr': 2e-5, 'weight_decay': 0.01}
}

# 3. 直接创建多GPU训练器
trainer = MultiGPUTrainer(
    model=model,
    num_gpus=4,
    optimizer_config=optimizer_config,
    aggregation_method=AggregationMethod.MEAN
)

# 4. 训练
trainer.setup()
epoch_stats = trainer.train_epoch(dataloader)
```

### 高级配置
```python
from unsloth_multigpu.utils import ConfigManager, DeviceManager

# 1. 设备管理
device_manager = DeviceManager()
devices = device_manager.get_available_devices()

# 2. 获取最优配置
config_manager = ConfigManager()
optimal_config = config_manager.get_optimal_config(
    model_size="7B",
    available_memory="32GB"
)

# 3. 启用多GPU（使用优化配置）
unsloth_multigpu.enable_multi_gpu(**optimal_config)
```

## 🛠️ 核心功能

### 1. 多GPU管理
- 自动检测和配置GPU设备
- 智能负载均衡
- 内存使用优化

### 2. 批次分片
- 支持不均匀分片
- 自适应分片策略
- 高效结果收集

### 3. DDP训练支持
- PyTorch原生DDP实现
- 自动梯度同步
- 高效的NCCL通信

### 4. 内存管理
- 实时内存监控
- OOM预防机制
- 自动内存清理

### 5. 配置管理
- 自动环境检测
- 最优配置生成
- 配置验证和模板

## 📊 性能特性

- **训练速度提升**: 支持3.5-4倍速度提升（4GPU环境）
- **内存优化**: 智能内存管理，降低OOM风险
- **稳定性**: 完整的错误处理和恢复机制
- **监控**: 实时性能监控和日志记录

## 🧪 测试

运行完整测试套件：
```bash
cd unsloth_multigpu
python tests/run_all_tests.py
```

运行快速测试：
```bash
python tests/run_all_tests.py --quick
```

## 📖 示例

查看 `examples/` 目录中的示例：
- `quick_start.py`: Hook机制基础示例（零侵入性）
- `advanced_config.py`: Hook机制高级配置示例  
- `direct_trainer_usage.py`: 直接使用MultiGPUTrainer示例
- `grpo_open_r1_multi_gpu.py`: GRPO强化学习多GPU训练示例
- `configs/grpo_open_r1_config.yaml`: GRPO训练YAML配置示例
- `verify_installation.py`: 安装验证脚本

### 选择合适的方式
- **现有项目迁移**: 使用 `quick_start.py` 的Hook方式
- **新项目开发**: 使用 `direct_trainer_usage.py` 的直接方式
- **GRPO强化学习训练**: 使用 `grpo_open_r1_multi_gpu.py` 进行多GPU GRPO训练
- **高级配置**: 参考 `advanced_config.py`

## ⚠️ 注意事项

1. **依赖关系**: 确保先安装Unsloth包
2. **CUDA环境**: 需要CUDA 11.0+支持  
3. **内存要求**: 建议每个GPU至少8GB显存
4. **Python版本**: 需要Python 3.8+
5. **DDP训练**: 多GPU训练需要使用torchrun启动
6. **显存优化**: 建议使用 `load_in_4bit=True` 减少显存占用
7. **GRPO训练**: 使用vLLM时需要先启动推理服务，支持open-r1专业奖励函数
8. **vLLM服务**: GRPO训练前需要启动vLLM服务用于快速推理

## 🤝 兼容性

- ✅ Unsloth 2023.x 版本
- ✅ PyTorch 1.12+
- ✅ CUDA 11.0+
- ✅ Transformers 4.30+

## 📝 更新日志

### v1.0.0 (2025-06)
- 完整的多GPU训练支持
- 模块化Hook系统
- 内存管理和优化
- 完整测试覆盖

**注意**: 此包为Unsloth的外部扩展，不会修改Unsloth源码，确保与上游版本的兼容性。 