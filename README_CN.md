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
- Unsloth (包含 unsloth_train 函数)
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
# 使用2个GPU运行快速开始示例
CUDA_VISIBLE_DEVICES=0,1 python examples/quick_start.py
```

#### 代码示例
```python
import unsloth_multigpu as unsloth_multigpu
from unsloth import FastLanguageModel, unsloth_train

# 1. 启用多GPU支持（Hook机制）
unsloth_multigpu.enable_multi_gpu(
    num_gpus=2,  # 使用2个GPU
    batch_size_per_gpu=2,  # 每个GPU的批次大小
    gradient_aggregation="mean"  # 梯度聚合策略
)

# 2. 加载模型（自动支持多GPU）
model, tokenizer = FastLanguageModel.from_pretrained(
    "/path/to/your/model",  # 模型路径
    max_seq_length=4096,
    dtype=torch.bfloat16,  # 注意：使用torch.bfloat16而不是字符串
    load_in_4bit=True
)

# 3. 使用原生unsloth_train（内部被Hook替换为多GPU逻辑）
trainer_stats = unsloth_train(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    # ... 其他训练参数
)
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

## 📁 项目结构

```
unsloth_multigpu/
├── __init__.py              # 主入口
├── core/                    # 核心组件
│   ├── multi_gpu_manager.py # 多GPU管理器
│   ├── batch_sharding.py    # 批次分片器
│   ├── gradient_aggregator.py # 梯度聚合器
│   ├── multi_gpu_trainer.py # 多GPU训练器
│   └── memory_manager.py    # 内存管理器
├── hooks/                   # Hook系统
│   ├── training_hooks.py    # 训练Hook
│   ├── loader_hooks.py      # 加载Hook
│   └── trainer_hooks.py     # 训练器Hook
├── utils/                   # 工具模块
│   ├── device_utils.py      # 设备管理
│   ├── logging_utils.py     # 日志系统
│   └── config_utils.py      # 配置管理
├── examples/                # 示例代码
│   ├── quick_start.py       # 快速开始
│   └── advanced_config.py   # 高级配置
└── tests/                   # 测试套件
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

### 3. 梯度聚合
- 多种聚合策略（平均、加权、中位数）
- 梯度一致性验证
- 数值稳定性保证

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
- `verify_installation.py`: 安装验证脚本

### 选择合适的方式
- **现有项目迁移**: 使用 `quick_start.py` 的Hook方式
- **新项目开发**: 使用 `direct_trainer_usage.py` 的直接方式
- **高级配置**: 参考 `advanced_config.py`

## ⚠️ 注意事项

1. **依赖关系**: 确保先安装Unsloth包
2. **CUDA环境**: 需要CUDA 11.0+支持
3. **内存要求**: 建议每个GPU至少8GB显存
4. **Python版本**: 需要Python 3.8+

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