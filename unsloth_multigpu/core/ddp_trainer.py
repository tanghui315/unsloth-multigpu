"""
DDP Trainer - Distributed Data Parallel Trainer based on PyTorch DDP
Replaces the original inefficient serial training implementation
"""

import logging
import time
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from transformers import Trainer

from .ddp_manager import DDPManager
from ..utils import MultiGPUConfig

logger = logging.getLogger(__name__)


class DDPTrainer:
    """
    DDP Trainer - Efficient distributed trainer based on native PyTorch DDP
    
    Main advantages:
    1. True parallel training (vs original serial)
    2. Automatic gradient synchronization (vs manual CPU transfer)
    3. Efficient NCCL communication (vs inefficient aggregation)
    4. Native PyTorch support (vs custom implementation)
    """
    
    def __init__(self, 
                 original_trainer: Trainer,
                 config: MultiGPUConfig,
                 ddp_manager: Optional[DDPManager] = None):
        """
        Initialize DDP Trainer
        
        Args:
            original_trainer: Original HuggingFace Trainer
            config: Multi-GPU configuration
            ddp_manager: DDP manager, None means auto-create
        """
        self.original_trainer = original_trainer
        self.config = config
        self.ddp_manager = ddp_manager or DDPManager()
        
        # Model and optimizer
        self.model = original_trainer.model
        self.ddp_model = None
        self.optimizer = None
        
        # Data related
        self.train_dataset = original_trainer.train_dataset
        self.data_collator = original_trainer.data_collator
        self.train_dataloader = None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.is_setup = False
        
        # Performance stats
        self.stats = {
            'total_train_time': 0.0,
            'total_forward_time': 0.0,
            'total_backward_time': 0.0,
            'total_samples': 0,
            'steps_per_second': 0.0
        }
        
        logger.info(f"🔧 Initialize DDPTrainer: {config.num_gpus} GPUs")
    
    def setup(self, rank: int):
        """
        Setup DDP training environment
        
        Args:
            rank: Current process rank
        """
        if self.is_setup:
            return
            
        logger.info(f"🚀 Setting up DDP training environment (rank={rank})...")
        
        # 1. Initialize DDP process group
        success = self.ddp_manager.init_process_group(
            rank=rank,
            world_size=self.config.num_gpus
        )
        if not success:
            raise RuntimeError(f"Failed to initialize DDP process group (rank={rank})")
        
        # 2. Wrap model with DDP
        self.ddp_model = self.ddp_manager.wrap_model(
            self.model,
            find_unused_parameters=getattr(self.config, 'find_unused_parameters', False)
        )
        
        # 3. Create optimizer (must be after DDP wrapping)
        self._setup_optimizer()
        
        # 4. Create distributed dataloader
        self._setup_dataloader()
        
        self.is_setup = True
        
        if self.ddp_manager.is_master:
            logger.info("✅ DDP training environment setup complete")
    
    def _setup_optimizer(self):
        """Setup optimizer"""
        # Extract optimizer config from original trainer
        training_args = self.original_trainer.args
        
        # Create optimizer
        if hasattr(self.original_trainer, 'create_optimizer'):
            # Use HuggingFace optimizer creation method
            self.original_trainer.model = self.ddp_model
            self.original_trainer.create_optimizer()
            self.optimizer = self.original_trainer.optimizer
        else:
            # Fallback to default AdamW
            self.optimizer = torch.optim.AdamW(
                self.ddp_model.parameters(),
                lr=training_args.learning_rate,
                weight_decay=training_args.weight_decay,
                eps=training_args.adam_epsilon
            )
        
        if self.ddp_manager.is_master:
            logger.info(f"✅ Optimizer setup complete: {type(self.optimizer).__name__}")
    
    def _setup_dataloader(self):
        """Setup distributed dataloader"""
        # Create distributed sampler
        sampler = self.ddp_manager.create_data_sampler(
            self.train_dataset,
            shuffle=True
        )
        
        # Create DataLoader
        self.train_dataloader = DataLoader(
            self.train_dataset,
            sampler=sampler,
            batch_size=self.config.batch_size_per_gpu,
            collate_fn=self.data_collator,
            num_workers=getattr(self.config, 'dataloader_num_workers', 0),
            pin_memory=True
        )
        
        if self.ddp_manager.is_master:
            logger.info(f"✅ Distributed dataloader setup complete: batch_size={self.config.batch_size_per_gpu}")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Execute one training step
        
        Args:
            batch: Training batch data
            
        Returns:
            Dict: Training result statistics
        """
        step_start = time.time()
        
        # Move data to GPU
        device = f"cuda:{self.ddp_manager.local_rank}"
        batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        forward_start = time.time()
        
        # Support mixed precision training
        if getattr(self.config, 'fp16', False):
            with torch.cuda.amp.autocast():
                outputs = self.ddp_model(**batch)
                loss = outputs.loss
        else:
            outputs = self.ddp_model(**batch)
            loss = outputs.loss
        
        forward_time = time.time() - forward_start
        
        # Backward pass (DDP automatically synchronizes gradients)
        backward_start = time.time()
        loss.backward()
        backward_time = time.time() - backward_start
        
        # Gradient clipping
        if getattr(self.config, 'max_grad_norm', None):
            torch.nn.utils.clip_grad_norm_(
                self.ddp_model.parameters(), 
                self.config.max_grad_norm
            )
        
        # Optimizer step
        self.optimizer.step()
        
        # Update global step
        self.global_step += 1
        
        # Statistics
        step_time = time.time() - step_start
        batch_size = batch['input_ids'].size(0)
        
        self.stats['total_forward_time'] += forward_time
        self.stats['total_backward_time'] += backward_time
        self.stats['total_samples'] += batch_size
        
        return {
            'loss': loss.item(),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'step_time': step_time,
            'forward_time': forward_time,
            'backward_time': backward_time,
            'throughput': batch_size / step_time,
            'global_step': self.global_step
        }
    
    def train_epoch(self) -> Dict[str, Any]:
        """
        Train one epoch
        
        Returns:
            Dict: Epoch training statistics
        """
        if not self.is_setup:
            raise RuntimeError("DDPTrainer not set up, please call setup() first")
        
        epoch_start = time.time()
        epoch_loss = 0.0
        num_steps = 0
        
        # Set sampler epoch (ensure different data order each epoch)
        if hasattr(self.train_dataloader.sampler, 'set_epoch'):
            self.train_dataloader.sampler.set_epoch(self.epoch)
        
        self.ddp_model.train()
        
        if self.ddp_manager.is_master:
            logger.info(f"🚀 Start training epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            step_result = self.train_step(batch)
            epoch_loss += step_result['loss']
            num_steps += 1
            
            # Periodically print progress (master process only)
            if self.ddp_manager.is_master and batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {self.epoch}, Step {batch_idx}/{len(self.train_dataloader)}, "
                    f"Loss: {step_result['loss']:.4f}, "
                    f"LR: {step_result['learning_rate']:.2e}, "
                    f"Throughput: {step_result['throughput']:.1f} samples/s"
                )
        
        # Epoch statistics
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / num_steps if num_steps > 0 else 0.0
        
        self.epoch += 1
        self.stats['total_train_time'] += epoch_time
        self.stats['steps_per_second'] = num_steps / epoch_time
        
        epoch_stats = {
            'epoch': self.epoch - 1,
            'avg_loss': avg_loss,
            'epoch_time': epoch_time,
            'num_steps': num_steps,
            'steps_per_second': self.stats['steps_per_second']
        }
        
        if self.ddp_manager.is_master:
            logger.info(f"✅ Epoch {self.epoch-1} finished: "
                       f"avg_loss={avg_loss:.4f}, "
                       f"time={epoch_time:.1f}s, "
                       f"steps/s={epoch_stats['steps_per_second']:.1f}")
        
        return epoch_stats
    
    def train(self, num_epochs: Optional[int] = None) -> Dict[str, Any]:
        """
        Run full training
        
        Args:
            num_epochs: Number of epochs, None means use config value
            
        Returns:
            Dict: Training result statistics
        """
        if num_epochs is None:
            num_epochs = getattr(self.config, 'num_train_epochs', 3)
        
        train_start = time.time()
        all_epoch_stats = []
        
        if self.ddp_manager.is_master:
            logger.info(f"🚀 Start DDP training: {num_epochs} epochs, {self.config.num_gpus} GPUs")
        
        try:
            for epoch_idx in range(num_epochs):
                epoch_stats = self.train_epoch()
                all_epoch_stats.append(epoch_stats)
            
            # Final statistics
            total_time = time.time() - train_start
            final_stats = {
                'total_time': total_time,
                'num_epochs': num_epochs,
                'total_steps': self.global_step,
                'avg_steps_per_second': self.global_step / total_time,
                'total_samples': self.stats['total_samples'],
                'epoch_stats': all_epoch_stats,
                'performance_stats': self.stats
            }
            
            if self.ddp_manager.is_master:
                logger.info(f"🎉 DDP training finished! "
                           f"Total time: {total_time:.1f}s, "
                           f"Total steps: {self.global_step}, "
                           f"Average speed: {final_stats['avg_steps_per_second']:.1f} steps/s")
            
            return final_stats
            
        except Exception as e:
            logger.error(f"❌ Error during DDP training: {e}")
            raise
        finally:
            # Cleanup resources
            self.cleanup()
    
    def cleanup(self):
        """Cleanup DDP resources"""
        try:
            self.ddp_manager.cleanup()
            if self.ddp_manager.is_master:
                logger.info("✅ DDP resources cleanup complete")
        except Exception as e:
            logger.warning(f"⚠️ Warning during DDP cleanup: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            'ddp_stats': self.ddp_manager.get_status(),
            'training_stats': self.stats.copy(),
            'config': {
                'num_gpus': self.config.num_gpus,
                'batch_size_per_gpu': self.config.batch_size_per_gpu,
                'ddp_backend': getattr(self.config, 'ddp_backend', 'nccl')
            }
        }
    
    def __del__(self):
        """Destructor"""
        try:
            self.cleanup()
        except:
            pass