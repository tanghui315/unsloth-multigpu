"""
Unsloth Multi-GPU Core 模块
包含多GPU训练的核心组件
"""

from .batch_sharding import AdaptiveBatchSharding, BatchSharding
from .gradient_aggregator import AggregationMethod, GradientAggregator
from .memory_manager import MemoryManager
from .multi_gpu_manager import MultiGPUManager
from .multi_gpu_trainer import MultiGPUTrainer

__all__ = [
    'MultiGPUManager',
    'MultiGPUTrainer',
    'BatchSharding', 
    'AdaptiveBatchSharding',
    'GradientAggregator',
    'AggregationMethod',
    'MemoryManager'
]

# 版本信息
__version__ = "1.0.0"
