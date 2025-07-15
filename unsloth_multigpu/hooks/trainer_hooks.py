"""
Trainer related Hook implementation
Responsible for handling Trainer configuration, callback functions, and performance monitoring, etc.
"""

import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

# Import DDP-aware logging
from ..utils.ddp_logging import get_ddp_logger

logger = get_ddp_logger(__name__)


class TrainerHooks:
    """
    Trainer related Hook manager
    
    Responsible:
    1. Trainer configuration multi-GPU optimization
    2. Trainer callback function Hook
    3. Performance monitoring and statistics
    4. Trainer state management
    """
    
    def __init__(self):
        """Initialize TrainerHooks manager"""
        self.original_functions = {}
        self.hooks_applied = False
        self.performance_stats = {
            'hook_overhead_ms': 0.0,
            'total_hook_calls': 0,
            'successful_hooks': 0,
            'failed_hooks': 0
        }
        
        logger.info("🔧 Initialize TrainerHooks manager")
    
    def apply_hooks(self) -> bool:
        """Apply all Trainer Hooks"""
        try:
            success = True
            
            # Apply Trainer initialization Hook
            if not self._hook_trainer_init():
                success = False
            
            # Apply train Hook, use MultiGPUTrainer to take over training loop
            if not self._hook_train():
                success = False
            
            if success:
                self.hooks_applied = True
                logger.info("✅ All Trainer Hooks applied successfully")
            else:
                logger.error("❌ Some Trainer Hooks failed to apply")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Failed to apply all Trainer Hooks: {e}")
            return False
    
    def remove_hooks(self) -> bool:
        """
        Remove all Trainer Hooks, restore original functions
        
        Returns:
            bool: Whether the Hooks were successfully removed
        """
        if not self.hooks_applied:
            logger.warning("Trainer Hooks not applied, no need to remove")
            return True
        
        success = True
        
        for func_name, func_info in self.original_functions.items():
            try:
                setattr(func_info['module'], func_info['attr'], func_info['original'])
                logger.info(f"✅ Restored function: {func_name}")
            except Exception as e:
                logger.error(f"❌ Failed to restore function {func_name}: {e}")
                success = False
        
        if success:
            self.hooks_applied = False
            self.original_functions.clear()
            logger.info("✅ All Trainer Hooks removed successfully")
        
        return success
    
    def _hook_trainer_init(self) -> bool:
        """
        Hook Trainer initialization, add multi-GPU configuration
        
        Returns:
            bool: Whether the Hook was successfully applied
        """
        try:
            # Try to import transformers
            from transformers import Trainer

            # Save original __init__ function
            original_init = Trainer.__init__
            self.original_functions['trainer_init'] = {
                'module': Trainer,
                'attr': '__init__',
                'original': original_init
            }
            
            def patched_trainer_init(self, *args, **kwargs):
                """Support multi-GPU Trainer initialization"""
                # Call original initialization
                result = original_init(self, *args, **kwargs)
                
                # Add multi-GPU configuration
                if os.environ.get("UNSLOTH_MULTIGPU_ENABLED") == "1":
                    # If Trainer instance does not already contain this method, dynamically inject
                    if not hasattr(self, "_setup_multi_gpu_config"):
                        def _setup_multi_gpu_config(trainer_self):
                            """Set multi-GPU related attributes for Trainer instance (dynamically injected)"""
                            trainer_self._unsloth_multi_gpu_enabled = True
                            trainer_self._unsloth_num_gpus = int(os.environ.get("UNSLOTH_MULTIGPU_NUM_GPUS", "1"))
                            trainer_self._unsloth_performance_stats = {
                                'multi_gpu_steps': 0,
                                'total_multi_gpu_time': 0.0,
                                'avg_step_time': 0.0
                            }
                            logger.info(f"🔧 Trainer multi-GPU configuration completed: {trainer_self._unsloth_num_gpus} GPUs")
                        # Bind to current instance and its class, ensuring it's available later
                        setattr(self.__class__, "_setup_multi_gpu_config", _setup_multi_gpu_config)
                    # Call injected method
                    self._setup_multi_gpu_config()
                
                return result
            
            # Replace function
            Trainer.__init__ = patched_trainer_init
            logger.info("✅ Hook Trainer.__init__ applied successfully")
            return True
            
        except ImportError:
            logger.warning("⚠️ transformers.Trainer not found, skipping Trainer initialization Hook")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to hook Trainer initialization: {e}")
            return False
    
    def _hook_train(self) -> bool:
        """
        Hook Trainer.train, make it call real DDP launcher for multi-GPU training
        """
        try:
            from transformers import Trainer

            import unsloth_multigpu as uns_mgpu
            from unsloth_multigpu.core import launch_ddp_training

            # Save original train method
            original_train = Trainer.train
            self.original_functions['train'] = {
                'module': Trainer,
                'attr': 'train',
                'original': original_train
            }

            def patched_train(self, *args, **kwargs):
                """Patched train method, supports multi-GPU DDP"""
                # If multi-GPU is not enabled, fall back to original method
                if not uns_mgpu.is_multi_gpu_enabled():
                    return original_train(self, *args, **kwargs)

                # Get active configuration
                cfg = uns_mgpu.get_active_config()
                
                # Check if should use DDP (multi-process training)
                # Handle both MultiGPUConfig object and dict fallback
                num_gpus = 1
                if cfg:
                    if hasattr(cfg, 'num_gpus'):
                        num_gpus = cfg.num_gpus
                    elif isinstance(cfg, dict) and 'num_gpus' in cfg:
                        num_gpus = cfg['num_gpus']
                
                if cfg and num_gpus > 1:
                    logger.info(f"🚀 Launching real DDP training: {num_gpus} GPUs")
                    
                    # Use a simpler solution: check if already in DDP environment
                    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
                        # Already in DDP environment, use DDP wrapper
                        logger.info("🔧 Detected DDP environment, training in DDP mode")
                        return _train_with_ddp_wrapper(self, original_train, *args, **kwargs)
                    else:
                        # Not in DDP environment, prompt user to use torchrun
                        logger.error("❌ Multi-GPU training requires launching with torchrun")
                        logger.info("💡 Please use the following command to launch:")
                        logger.info(f"   CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node={num_gpus} your_script.py")
                        logger.info("")
                        logger.info("⚠️ GPU memory optimization suggestions:")
                        logger.info("   1. Use load_in_4bit=True to reduce model memory usage")
                        logger.info("   2. Reduce batch_size_per_gpu")
                        logger.info("   3. Enable gradient_checkpointing=True")
                        raise RuntimeError("Multi-GPU training requires launching with torchrun, please refer to the above command")
                else:
                    # Single GPU or fallback
                    logger.info("🔄 Single GPU mode, using original training")
                    return original_train(self, *args, **kwargs)

            # Replace function
            Trainer.train = patched_train
            logger.info("✅ Hook Trainer.train applied successfully")
            return True
        except ImportError:
            logger.warning("⚠️ transformers.Trainer not found, skipping train Hook")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to hook train: {e}")
            return False


def _train_with_ddp_wrapper(trainer, original_train, *args, **kwargs):
    """
    Wrapper function for training in DDP environment
    
    Args:
        trainer: Trainer instance
        original_train: Original train function
        *args, **kwargs: Training parameters
        
    Returns:
        Training result
    """
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    
    try:
        # Initialize DDP process group (if not already initialized)
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        
        rank = dist.get_rank()
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        world_size = dist.get_world_size()
        
        logger.info_rank0(f"🚀 DDP training started: {world_size} processes")
        logger.info_all_ranks(f"Process started (local GPU {local_rank})")
        
        # Set CUDA device
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        
        # Move model to corresponding GPU and wrap as DDP
        if not isinstance(trainer.model, DDP):
            # GPU memory optimization: clear unnecessary cache
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            
            # Ensure model is on the correct device
            if trainer.model.device != device:
                logger.info_rank0(f"🔄 Moving model to target devices")
                trainer.model = trainer.model.to(device)
            
            # Wrap as DDP, enable gradient compression to reduce communication overhead
            trainer.model = DDP(
                trainer.model, 
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,  # Reduce memory usage
                static_graph=True  # Optimize if computation graph is static
            )
            logger.info_rank0(f"✅ Models wrapped as DDP across {world_size} GPUs")
            
            # GPU memory statistics
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(local_rank) / 1024**3
                cached = torch.cuda.memory_reserved(local_rank) / 1024**3
                logger.info_rank0(f"📊 Memory per GPU: ~{allocated:.2f}GB allocated, {cached:.2f}GB cached")
                logger.info_rank0(f"📊 Total DDP memory: ~{allocated * world_size:.2f}GB across {world_size} GPUs")
        
        # Set distributed data sampler
        from torch.utils.data.distributed import DistributedSampler
        if trainer.train_dataset is not None:
            # Check if DistributedSampler is already set
            if not hasattr(trainer, '_ddp_sampler_set'):
                total_samples = len(trainer.train_dataset)
                sampler = DistributedSampler(
                    trainer.train_dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True
                )
                trainer.args.dataloader_sampler = sampler
                trainer._ddp_sampler_set = True
                
                # Log data split info
                logger.info_rank0("✅ Distributed data sampler configured")
                logger.info_rank0(f"📊 Total samples: {total_samples}, samples per GPU: ~{len(sampler)}")
                logger.info_all_ranks(f"Will process ~{len(sampler)} samples per epoch")
        
        # Execute original training
        result = original_train(trainer, *args, **kwargs)
        
        logger.info_rank0("✅ DDP training completed successfully")
        
        return result
        
    except Exception as e:
        logger.error_rank0(f"❌ DDP training failed: {e}")
        raise

