"""
DDP Launcher - Distributed Training Launcher
Based on torch.multiprocessing for multi-process DDP training
"""

import logging
import os
import sys
from typing import Any, Callable, Dict, Optional

import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Process

from .ddp_manager import setup_ddp_environment
from .ddp_trainer import DDPTrainer
from ..utils import MultiGPUConfig

logger = logging.getLogger(__name__)


class DDPLauncher:
    """
    DDP Launcher - Distributed Training Launcher
    
    Main functions:
    1. Manage multi-process training launch
    2. Error handling and process monitoring
    3. Result collection and aggregation
    4. Resource cleanup
    """
    
    def __init__(self, config: MultiGPUConfig):
        """
        Initialize DDP Launcher
        
        Args:
            config: Multi-GPU configuration
        """
        self.config = config
        self.processes = []
        self.results = {}
        
        logger.info(f"🔧 Initializing DDPLauncher: {config.num_gpus} GPUs")
    
    def launch_training(self, 
                       trainer_fn: Callable,
                       trainer_args: tuple = (),
                       trainer_kwargs: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Launch distributed training
        
        Args:
            trainer_fn: Training function
            trainer_args: Positional arguments for training function
            trainer_kwargs: Keyword arguments for training function
            
        Returns:
            Dict: Training results
        """
        if trainer_kwargs is None:
            trainer_kwargs = {}
            
        logger.info(f"🚀 Launching DDP training: {self.config.num_gpus} processes")
        
        try:
            # Set up DDP environment
            setup_ddp_environment(self.config.num_gpus)
            
            # Use spawn method to start processes (recommended for CUDA)
            mp.set_start_method('spawn', force=True)
            
            # Create processes
            processes = []
            for rank in range(self.config.num_gpus):
                p = Process(
                    target=self._worker_process,
                    args=(rank, trainer_fn, trainer_args, trainer_kwargs)
                )
                p.start()
                processes.append(p)
            
            # Wait for all processes to finish
            for p in processes:
                p.join()
            
            # Check process status
            success_count = 0
            for rank, p in enumerate(processes):
                if p.exitcode == 0:
                    success_count += 1
                else:
                    logger.error(f"❌ Process {rank} exited abnormally: exit_code={p.exitcode}")
            
            if success_count == self.config.num_gpus:
                logger.info("✅ All DDP processes completed successfully")
                return {'status': 'success', 'num_processes': success_count}
            else:
                logger.error(f"❌ DDP training failed: {success_count}/{self.config.num_gpus} processes succeeded")
                return {'status': 'failed', 'success_count': success_count}
                
        except Exception as e:
            logger.error(f"❌ DDP launch failed: {e}")
            self._cleanup_processes()
            raise
    
    def _worker_process(self, 
                       rank: int, 
                       trainer_fn: Callable,
                       trainer_args: tuple,
                       trainer_kwargs: Dict[str, Any]):
        """
        Worker process function
        
        Args:
            rank: Process rank
            trainer_fn: Training function
            trainer_args: Training function arguments
            trainer_kwargs: Training function keyword arguments
        """
        try:
            # Set process logger
            process_logger = logging.getLogger(f"ddp_worker_{rank}")
            process_logger.info(f"🚀 Starting DDP worker process (rank={rank})")
            
            # Call training function
            result = trainer_fn(rank, *trainer_args, **trainer_kwargs)
            
            process_logger.info(f"✅ DDP worker process completed (rank={rank})")
            return result
            
        except Exception as e:
            logger.error(f"❌ DDP worker process failed (rank={rank}): {e}")
            logger.error(f"Exception details:", exc_info=True)
            sys.exit(1)
    
    def _cleanup_processes(self):
        """Clean up process resources"""
        for p in self.processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5.0)
                if p.is_alive():
                    p.kill()
        
        self.processes.clear()
        logger.info("🧹 Process resources cleaned up")


def ddp_train_worker(rank: int, 
                    original_trainer,
                    config: MultiGPUConfig) -> Dict[str, Any]:
    """
    DDP training worker function
    
    Args:
        rank: Process rank
        original_trainer: Original trainer
        config: Multi-GPU configuration
        
    Returns:
        Dict: Training results
    """
    from .ddp_manager import DDPManager
    
    try:
        # Create DDP manager and trainer
        ddp_manager = DDPManager()
        ddp_trainer = DDPTrainer(original_trainer, config, ddp_manager)
        
        # Set up DDP environment
        ddp_trainer.setup(rank)
        
        # Run training
        result = ddp_trainer.train()
        
        return result
        
    except Exception as e:
        logger.error(f"❌ DDP training worker process failed (rank={rank}): {e}")
        raise


def launch_ddp_training(original_trainer, config: MultiGPUConfig) -> Dict[str, Any]:
    """
    Convenient DDP training launch function
    
    Args:
        original_trainer: Original HuggingFace trainer
        config: Multi-GPU configuration
        
    Returns:
        Dict: Training results
    """
    launcher = DDPLauncher(config)
    
    return launcher.launch_training(
        trainer_fn=ddp_train_worker,
        trainer_args=(original_trainer, config)
    )