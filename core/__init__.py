"""
Unsloth Multi-GPU Core module
Contains the core components for multi-GPU training
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

# Version information
__version__ = "1.0.0"
