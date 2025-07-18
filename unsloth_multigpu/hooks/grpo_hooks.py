"""
GRPO (Generalized Reinforcement Learning from Policy Optimization) hooks
Extends Hook system to support reinforcement learning training
"""

import os
from typing import Any, Dict, List, Optional, Callable
from transformers import TrainerCallback, TrainerControl, TrainerState

from ..utils.ddp_logging import get_ddp_logger

logger = get_ddp_logger(__name__)


class GRPOHooks:
    """
    GRPO training hooks manager
    Provides GRPO-specific extensions to the existing Hook system
    """
    
    def __init__(self):
        self.hooks_applied = False
        self.original_functions = {}
        self.grpo_config = None
        
    def apply_hooks(self) -> bool:
        """Apply GRPO-specific hooks"""
        try:
            # Hook GRPOTrainer
            success = self._hook_grpo_trainer()
            
            if success:
                self.hooks_applied = True
                logger.info_rank0("✅ GRPO hooks applied successfully")
            else:
                logger.error_rank0("❌ Some GRPO hooks failed to apply")
            
            return success
            
        except Exception as e:
            logger.error_rank0(f"❌ Failed to apply GRPO hooks: {e}")
            return False
    
    def remove_hooks(self) -> bool:
        """Remove GRPO hooks and restore original functions"""
        if not self.hooks_applied:
            return True
            
        try:
            # Restore original functions
            for func_name, func_info in self.original_functions.items():
                setattr(func_info['module'], func_info['attr'], func_info['original'])
                logger.info_rank0(f"✅ Restored function: {func_name}")
            
            self.hooks_applied = False
            self.original_functions.clear()
            logger.info_rank0("✅ GRPO hooks removed successfully")
            return True
            
        except Exception as e:
            logger.error_rank0(f"❌ Failed to remove GRPO hooks: {e}")
            return False
    
    def _hook_grpo_trainer(self) -> bool:
        """Hook GRPOTrainer to support multi-GPU"""
        try:
            from trl import GRPOTrainer
            
            # Save original train method
            original_train = GRPOTrainer.train
            self.original_functions['grpo_train'] = {
                'module': GRPOTrainer,
                'attr': 'train',
                'original': original_train
            }
            
            def patched_grpo_train(self, *args, **kwargs):
                """Patched GRPO train method with multi-GPU support"""
                # Import here to avoid circular imports
                import unsloth_multigpu as ump
                
                # Check if multi-GPU is enabled
                if not ump.is_multi_gpu_enabled():
                    logger.info_rank0("🔄 Single GPU GRPO training")
                    return original_train(self, *args, **kwargs)
                
                # Get active configuration
                cfg = ump.get_active_config()
                
                # Check if should use DDP
                num_gpus = 1
                if cfg:
                    if hasattr(cfg, 'num_gpus'):
                        num_gpus = cfg.num_gpus
                    elif isinstance(cfg, dict) and 'num_gpus' in cfg:
                        num_gpus = cfg['num_gpus']
                
                if num_gpus > 1:
                    # DDP environment check
                    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
                        logger.info_rank0("🚀 DDP GRPO training started")
                        return self._train_with_ddp_wrapper(original_train, *args, **kwargs)
                    else:
                        # Not in DDP environment, provide guidance
                        logger.error_rank0("❌ GRPO multi-GPU training requires torchrun")
                        logger.info_rank0("💡 Please use the following command:")
                        logger.info_rank0(f"   CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node={num_gpus} your_script.py")
                        logger.info_rank0("")
                        logger.info_rank0("⚠️ GRPO considerations:")
                        logger.info_rank0("   1. Reward functions will be computed on each GPU")
                        logger.info_rank0("   2. vLLM inference needs careful coordination")
                        logger.info_rank0("   3. Consider using smaller num_generations per GPU")
                        raise RuntimeError("GRPO multi-GPU training requires torchrun")
                else:
                    # Single GPU
                    logger.info_rank0("🔄 Single GPU GRPO training")
                    return original_train(self, *args, **kwargs)
            
            # Add DDP wrapper method to GRPOTrainer
            def _train_with_ddp_wrapper(self, original_train, *args, **kwargs):
                """DDP wrapper for GRPO training"""
                import torch
                import torch.distributed as dist
                from torch.nn.parallel import DistributedDataParallel as DDP
                
                try:
                    # Initialize DDP if needed
                    if not dist.is_initialized():
                        dist.init_process_group(backend='nccl')
                    
                    rank = dist.get_rank()
                    local_rank = int(os.environ.get('LOCAL_RANK', rank))
                    world_size = dist.get_world_size()
                    
                    logger.info_rank0(f"🚀 GRPO DDP training: {world_size} processes")
                    logger.info_all_ranks(f"GRPO process started (GPU {local_rank})")
                    
                    # Set CUDA device
                    torch.cuda.set_device(local_rank)
                    device = torch.device(f'cuda:{local_rank}')
                    
                    # Wrap model as DDP if not already wrapped
                    if not isinstance(self.model, DDP):
                        if self.model.device != device:
                            self.model = self.model.to(device)
                        
                        self.model = DDP(
                            self.model,
                            device_ids=[local_rank],
                            output_device=local_rank,
                            find_unused_parameters=False
                        )
                        logger.info_rank0("✅ GRPO model wrapped as DDP")
                    
                    # Setup distributed data sampler for GRPO
                    self._setup_grpo_distributed_data(rank, world_size)
                    
                    # Execute original training
                    result = original_train(self, *args, **kwargs)
                    
                    logger.info_rank0("✅ GRPO DDP training completed")
                    return result
                    
                except Exception as e:
                    logger.error_rank0(f"❌ GRPO DDP training failed: {e}")
                    raise
            
            # Attach methods to GRPOTrainer
            GRPOTrainer.train = patched_grpo_train
            GRPOTrainer._train_with_ddp_wrapper = _train_with_ddp_wrapper
            GRPOTrainer._setup_grpo_distributed_data = self._setup_grpo_distributed_data
            
            logger.info_rank0("✅ Hook GRPOTrainer.train applied successfully")
            return True
            
        except ImportError:
            logger.warning_rank0("⚠️ TRL GRPOTrainer not found, skipping GRPO hooks")
            return True  # Not fatal
        except Exception as e:
            logger.error_rank0(f"❌ Failed to hook GRPOTrainer: {e}")
            return False
    
    @staticmethod
    def _setup_grpo_distributed_data(self, rank: int, world_size: int):
        """Setup distributed data sampling for GRPO"""
        try:
            from torch.utils.data.distributed import DistributedSampler
            
            if self.train_dataset is not None:
                if not hasattr(self, '_grpo_ddp_sampler_set'):
                    total_samples = len(self.train_dataset)
                    
                    sampler = DistributedSampler(
                        self.train_dataset,
                        num_replicas=world_size,
                        rank=rank,
                        shuffle=True
                    )
                    
                    # For GRPO, we need to be careful about data distribution
                    # as reward computation happens per batch
                    if hasattr(self.args, 'dataloader_sampler'):
                        self.args.dataloader_sampler = sampler
                    
                    self._grpo_ddp_sampler_set = True
                    
                    logger.info_rank0("✅ GRPO distributed data sampler configured")
                    logger.info_rank0(f"📊 Total samples: {total_samples}, per GPU: ~{len(sampler)}")
                    logger.info_all_ranks(f"Will process ~{len(sampler)} samples for reward computation")
                    
        except Exception as e:
            logger.error_rank0(f"❌ Failed to setup GRPO distributed data: {e}")
            raise
    
    def get_hook_status(self) -> Dict[str, Any]:
        """Get GRPO hooks status"""
        return {
            'grpo_hooks_applied': self.hooks_applied,
            'active_hooks': list(self.original_functions.keys()),
            'supported_trainers': ['GRPOTrainer'],
            'features': [
                'Multi-GPU GRPO training',
                'Distributed reward computation', 
                'DDP model wrapping',
                'Distributed data sampling'
            ]
        }


# Global GRPO hooks manager
grpo_hooks = GRPOHooks()


def enable_grpo_support() -> bool:
    """
    Enable GRPO (Generalized Reinforcement Learning) support for multi-GPU
    
    Usage:
        import unsloth_multigpu as ump
        
        # Enable GRPO support
        ump.enable_grpo_support()
        
        # Enable multi-GPU
        ump.enable_multi_gpu(num_gpus=2)
        
        # Use GRPOTrainer normally
        trainer = GRPOTrainer(...)
        trainer.train()  # Will use DDP automatically
    
    Returns:
        bool: Whether GRPO support was successfully enabled
    """
    return grpo_hooks.apply_hooks()


def disable_grpo_support() -> bool:
    """Disable GRPO support"""
    return grpo_hooks.remove_hooks()


def get_grpo_status() -> Dict[str, Any]:
    """Get GRPO support status"""
    return grpo_hooks.get_hook_status()