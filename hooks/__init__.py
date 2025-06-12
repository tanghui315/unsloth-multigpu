"""
Unsloth Multi-GPU Hooks module
Contains all Hook implementations for dynamic function replacement (monkey patching)
"""

from .loader_hooks import LoaderHooks
from .trainer_hooks import TrainerHooks
from .training_hooks import TrainingHooks

__all__ = [
    'TrainingHooks',
    'LoaderHooks', 
    'TrainerHooks'
]

# 版本信息
__version__ = "1.0.0"
