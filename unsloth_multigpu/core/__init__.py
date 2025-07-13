"""
Unsloth Multi-GPU Core module
Contains DDP-based multi-GPU training components
"""

# DDP components (high-performance PyTorch native implementation)
from .ddp_manager import DDPManager
from .ddp_trainer import DDPTrainer
from .ddp_launcher import DDPLauncher, launch_ddp_training
from .memory_manager import MemoryManager

__all__ = [
    'DDPManager',
    'DDPTrainer',
    'DDPLauncher',
    'launch_ddp_training',
    'MemoryManager'
]

# Version information
__version__ = "1.0.0"
