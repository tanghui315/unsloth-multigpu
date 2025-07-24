"""
GRPO (Generalized Reinforcement Learning from Policy Optimization) hooks
Extends Hook system to support reinforcement learning training
"""

import os
from typing import Any, Dict, List, Optional, Callable
from transformers import TrainerCallback, TrainerControl, TrainerState

from ..utils.ddp_logging import get_ddp_logger

logger = get_ddp_logger(__name__)

def _patch_ddp_class_comprehensively():
    try:
        from torch.nn.parallel import DistributedDataParallel as DDP
        
        if hasattr(DDP, '_unsloth_multigpu_patched'):
            logger.info_rank0("🔄 DDP class already comprehensively patched, skipping")
            return True
        
        logger.info_rank0("🌟 Applying comprehensive DDP class patch...")
        
        methods_to_forward = [
            # PEFT/LoRA
            "disable_adapter", "enable_adapter",
            "set_adapter", "get_adapter", 
            "merge_adapter", "unmerge_adapter",
            "active_adapter",
            "gradient_checkpointing_enable",
            "gradient_checkpointing_disable",
            "generate", "sample", "forward",
            "eval", "train", "save_pretrained",
            "from_pretrained", "resize_token_embeddings"
        ]
        
        attributes_to_forward = [
            "config", "generation_config", "model_parallel",
            "device", "dtype", "training"
        ]
        
        patched_methods = []
        for method_name in methods_to_forward:
            if hasattr(DDP, method_name):
                continue
                
            def create_method_forwarder(method_name=method_name):
                def forward_method(self, *args, **kwargs):
                    logger.debug(f"🔧 DDP.{method_name} forwarding to module")
                    if hasattr(self.module, method_name):
                        return getattr(self.module, method_name)(*args, **kwargs)
                    else:
                        raise AttributeError(f"Module {type(self.module)} has no attribute '{method_name}'")
                return forward_method
            
            setattr(DDP, method_name, create_method_forwarder())
            patched_methods.append(method_name)
        
        # 2. 重写__getattr__方法实现智能属性转发
        original_getattr = DDP.__getattr__ if hasattr(DDP, '__getattr__') else None
        
        def smart_getattr(self, name):
            """智能属性转发：自动转发到module的属性"""
            # 先尝试原始__getattr__（如果存在）
            if original_getattr:
                try:
                    return original_getattr(self, name)
                except AttributeError:
                    pass
            
            # 转发到module
            if hasattr(self, 'module') and hasattr(self.module, name):
                attr = getattr(self.module, name)
                logger.debug(f"🔧 DDP.__getattr__ forwarding '{name}' to module")
                return attr
            
            # 如果都没有，抛出标准错误
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        # 设置新的__getattr__
        DDP.__getattr__ = smart_getattr
        
        # 3. 标记已打补丁
        DDP._unsloth_multigpu_patched = True
        
        logger.info_rank0(f"🌟 DDP class comprehensively patched:")
        logger.info_rank0(f"   ✅ {len(patched_methods)} methods forwarded: {patched_methods[:5]}...")
        logger.info_rank0(f"   ✅ Smart __getattr__ for automatic attribute forwarding")
        logger.info_rank0(f"   ✅ All future DDP instances will have complete compatibility")
        
        return True
        
    except Exception as e:
        logger.error_rank0(f"❌ Failed to patch DDP class comprehensively: {e}")
        import traceback
        logger.error_rank0(f"   Traceback: {traceback.format_exc()}")
        return False


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
            logger.info_rank0("🌟 Applying comprehensive DDP class patch...")
            ddp_patch_success = _patch_ddp_class_comprehensively()
            if ddp_patch_success:
                logger.info_rank0("✅ Comprehensive DDP class patch applied successfully")
            else:
                logger.warning_rank0("⚠️ Comprehensive DDP class patch failed, will rely on instance patches")
            
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
                        
                        # Store the original model before wrapping
                        original_model = self.model
                        
                        self.model = DDP(
                            self.model,
                            device_ids=[local_rank],
                            output_device=local_rank,
                            find_unused_parameters=False
                        )
                        logger.info_rank0("✅ GRPO model wrapped as DDP")
                        
                        # Add gradient checkpointing compatibility for DDP
                        logger.info_rank0("🔧 About to call DDP gradient checkpointing compatibility setup...")
                        GRPOHooks._setup_ddp_gradient_checkpointing_compatibility(self, original_model)
                        logger.info_rank0("✅ DDP gradient checkpointing compatibility setup call completed")
                    
                    # Setup distributed data sampler for GRPO
                    logger.info_rank0("🔧 About to call GRPO distributed data setup...")
                    GRPOHooks._setup_grpo_distributed_data(self, rank, world_size)
                    logger.info_rank0("✅ GRPO distributed data setup call completed")
                    
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
            GRPOTrainer._setup_grpo_distributed_data = GRPOHooks._setup_grpo_distributed_data
            GRPOTrainer._setup_ddp_gradient_checkpointing_compatibility = GRPOHooks._setup_ddp_gradient_checkpointing_compatibility
            
            logger.info_rank0("✅ Hook GRPOTrainer.train applied successfully")
            return True
            
        except ImportError:
            logger.warning_rank0("⚠️ TRL GRPOTrainer not found, skipping GRPO hooks")
            return True  # Not fatal
        except Exception as e:
            logger.error_rank0(f"❌ Failed to hook GRPOTrainer: {e}")
            return False
    
    @staticmethod
    def _setup_grpo_distributed_data(trainer, rank: int, world_size: int):
        """Setup distributed data sampling for GRPO with robust compatibility"""
        logger.info_rank0(f"🚀 GRPO distributed data setup called for trainer type: {type(trainer)}")
        try:
            from torch.utils.data.distributed import DistributedSampler
            
            if trainer.train_dataset is None:
                logger.warning_rank0("⚠️ No train_dataset found, skipping distributed sampler setup")
                return
                
            if hasattr(trainer, '_grpo_ddp_sampler_set'):
                logger.info_rank0("🔄 GRPO distributed sampler already configured")
                return
            
            total_samples = len(trainer.train_dataset)
            logger.info_rank0(f"📊 Setting up GRPO distributed sampling: {total_samples} total samples")
            
            # Create distributed sampler
            sampler = DistributedSampler(
                trainer.train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=False  # Important for GRPO to avoid losing data
            )
            
            # Try multiple ways to set the sampler (compatibility with different TRL versions)
            sampler_set = False
            
            # Method 1: Set via training args (most common)
            if hasattr(trainer.args, 'dataloader_sampler'):
                trainer.args.dataloader_sampler = sampler
                logger.info_rank0("✅ Sampler set via args.dataloader_sampler")
                sampler_set = True
            
            # Method 2: Set via train_sampler (alternative approach)
            elif hasattr(trainer.args, 'train_sampler'):
                trainer.args.train_sampler = sampler
                logger.info_rank0("✅ Sampler set via args.train_sampler")
                sampler_set = True
                
            # Method 3: Direct trainer attribute (fallback)
            elif hasattr(trainer, '_train_sampler'):
                trainer._train_sampler = sampler
                logger.info_rank0("✅ Sampler set via trainer._train_sampler")
                sampler_set = True
                
            # Method 4: Force override get_train_dataloader if needed
            if not sampler_set:
                logger.warning_rank0("⚠️ Could not set sampler via standard methods, trying override")
                try:
                    # Store the sampler for later use
                    trainer._grpo_distributed_sampler = sampler
                    
                    # Override get_train_dataloader if it exists
                    if hasattr(trainer, 'get_train_dataloader'):
                        
                        def get_train_dataloader_with_sampler():
                            from torch.utils.data import DataLoader
                            return DataLoader(
                                trainer.train_dataset,
                                batch_size=trainer.args.per_device_train_batch_size,
                                sampler=sampler,
                                collate_fn=getattr(trainer, 'data_collator', None),
                                drop_last=getattr(trainer.args, 'dataloader_drop_last', False),
                                num_workers=getattr(trainer.args, 'dataloader_num_workers', 0),
                                pin_memory=getattr(trainer.args, 'dataloader_pin_memory', True),
                            )
                        
                        trainer.get_train_dataloader = get_train_dataloader_with_sampler
                        logger.info_rank0("✅ Sampler set via get_train_dataloader override")
                        sampler_set = True
                except Exception as override_e:
                    logger.warning_rank0(f"⚠️ DataLoader override failed: {override_e}")
            
            if sampler_set:
                trainer._grpo_ddp_sampler_set = True
                samples_per_gpu = len(sampler)
                logger.info_rank0(f"✅ GRPO distributed data sampler configured successfully")
                logger.info_rank0(f"📊 Total samples: {total_samples}, per GPU: ~{samples_per_gpu}")
                logger.info_all_ranks(f"🎯 GPU {rank} will process ~{samples_per_gpu} samples for reward computation")
            else:
                logger.warning_rank0("⚠️ Could not set distributed sampler - training may duplicate data across GPUs")
                logger.warning_rank0("💡 This may affect GRPO reward computation accuracy in multi-GPU setup")
                
        except ImportError as e:
            logger.error_rank0(f"❌ Failed to import DistributedSampler: {e}")
            logger.warning_rank0("⚠️ Continuing without distributed sampling - may cause data duplication")
        except Exception as e:
            logger.error_rank0(f"❌ Failed to setup GRPO distributed data: {e}")
            logger.warning_rank0("⚠️ Continuing without distributed sampling - may cause data duplication")
            import traceback
            logger.error_rank0(f"   Exception traceback: {traceback.format_exc()}")
            # Don't raise - allow training to continue with potential data duplication
    
    @staticmethod
    def _setup_ddp_gradient_checkpointing_compatibility(trainer, original_model):
        """Setup gradient checkpointing and LoRA adapter compatibility for DDP wrapped models"""
        logger.info_rank0(f"🚀 DDP compatibility setup called for trainer type: {type(trainer)}, model type: {type(trainer.model)}")
        try:
            from torch.nn.parallel import DistributedDataParallel as DDP
            
            logger.info_rank0("🔧 Setting up DDP gradient checkpointing and LoRA adapter compatibility...")
            
            # Ensure the model has gradient checkpointing methods exposed through DDP
            if isinstance(trainer.model, DDP):
                logger.info_rank0(f"🔍 Model is DDP wrapped, original model type: {type(original_model)}")
                logger.info_rank0(f"🔍 Original model has disable_adapter: {hasattr(original_model, 'disable_adapter')}")
                logger.info_rank0(f"🔍 Original model has enable_adapter: {hasattr(original_model, 'enable_adapter')}")
                
                # Critical diagnosis: Check trainer.accelerator availability
                if hasattr(trainer, 'accelerator'):
                    logger.info_rank0(f"🔍 Trainer has accelerator: {type(trainer.accelerator)}")
                    try:
                        unwrapped_test = trainer.accelerator.unwrap_model(trainer.model, keep_fp32_wrapper=False)
                        logger.info_rank0(f"🔍 Accelerator unwrap test successful, unwrapped type: {type(unwrapped_test)}")
                        logger.info_rank0(f"🔍 Unwrapped test model has disable_adapter: {hasattr(unwrapped_test, 'disable_adapter')}")
                        logger.info_rank0(f"🔍 Unwrapped test model has enable_adapter: {hasattr(unwrapped_test, 'enable_adapter')}")
                    except Exception as test_e:
                        logger.error_rank0(f"❌ Accelerator unwrap test failed: {test_e}")
                        import traceback
                        logger.error_rank0(f"   Traceback: {traceback.format_exc()}")
                else:
                    logger.error_rank0("❌ Critical issue: Trainer does not have accelerator attribute!")
                    logger.error_rank0("   This explains why DDP method forwarding fails")
                    logger.error_rank0(f"   Available trainer attributes: {[attr for attr in dir(trainer) if not attr.startswith('_')][:10]}...")
                # Forward gradient checkpointing methods from the underlying module
                if hasattr(original_model, 'gradient_checkpointing_enable'):
                    def gradient_checkpointing_enable(*args, **kwargs):
                        logger.info_rank0("🔧 DDP gradient_checkpointing_enable called, unwrapping model with accelerator")
                        try:
                            unwrapped_model = trainer.accelerator.unwrap_model(trainer.model, keep_fp32_wrapper=False)
                            if hasattr(unwrapped_model, 'gradient_checkpointing_enable'):
                                result = unwrapped_model.gradient_checkpointing_enable(*args, **kwargs)
                                logger.info_rank0("✅ Successfully called gradient_checkpointing_enable via unwrap_model")
                                return result
                            else:
                                logger.info_rank0("🔄 Falling back to direct .module access")
                                return trainer.model.module.gradient_checkpointing_enable(*args, **kwargs)
                        except Exception as e:
                            logger.warning_rank0(f"⚠️ accelerator.unwrap_model failed: {e}")
                            logger.info_rank0("🔄 Falling back to direct .module access")
                            return trainer.model.module.gradient_checkpointing_enable(*args, **kwargs)
                    
                    # Add method to DDP wrapper
                    trainer.model.gradient_checkpointing_enable = gradient_checkpointing_enable
                    logger.info_rank0("✅ Added gradient_checkpointing_enable to DDP model")
                
                if hasattr(original_model, 'gradient_checkpointing_disable'):
                    def gradient_checkpointing_disable(*args, **kwargs):
                        logger.info_rank0("🔧 DDP gradient_checkpointing_disable called, unwrapping model with accelerator")
                        try:
                            unwrapped_model = trainer.accelerator.unwrap_model(trainer.model, keep_fp32_wrapper=False)
                            if hasattr(unwrapped_model, 'gradient_checkpointing_disable'):
                                result = unwrapped_model.gradient_checkpointing_disable(*args, **kwargs)
                                logger.info_rank0("✅ Successfully called gradient_checkpointing_disable via unwrap_model")
                                return result
                            else:
                                logger.info_rank0("🔄 Falling back to direct .module access")
                                return trainer.model.module.gradient_checkpointing_disable(*args, **kwargs)
                        except Exception as e:
                            logger.warning_rank0(f"⚠️ accelerator.unwrap_model failed: {e}")
                            logger.info_rank0("🔄 Falling back to direct .module access")
                            return trainer.model.module.gradient_checkpointing_disable(*args, **kwargs)
                    
                    # Add method to DDP wrapper
                    trainer.model.gradient_checkpointing_disable = gradient_checkpointing_disable
                    logger.info_rank0("✅ Added gradient_checkpointing_disable to DDP model")
                
                # Also forward the gradient_checkpointing property if it exists
                if hasattr(original_model, 'gradient_checkpointing'):
                    def get_gradient_checkpointing(self):
                        try:
                            unwrapped_model = trainer.accelerator.unwrap_model(trainer.model, keep_fp32_wrapper=False)
                            return getattr(unwrapped_model, 'gradient_checkpointing', False)
                        except Exception as e:
                            logger.warning_rank0(f"⚠️ accelerator.unwrap_model failed in gradient_checkpointing getter: {e}")
                            return getattr(trainer.model.module, 'gradient_checkpointing', False)
                    
                    def set_gradient_checkpointing(self, value):
                        try:
                            unwrapped_model = trainer.accelerator.unwrap_model(trainer.model, keep_fp32_wrapper=False)
                            if hasattr(unwrapped_model, 'gradient_checkpointing'):
                                unwrapped_model.gradient_checkpointing = value
                                logger.info_rank0(f"✅ Set gradient_checkpointing={value} via unwrap_model")
                            else:
                                logger.info_rank0("🔄 Falling back to direct .module access for gradient_checkpointing setter")
                                if hasattr(trainer.model.module, 'gradient_checkpointing'):
                                    trainer.model.module.gradient_checkpointing = value
                        except Exception as e:
                            logger.warning_rank0(f"⚠️ accelerator.unwrap_model failed in gradient_checkpointing setter: {e}")
                            if hasattr(trainer.model.module, 'gradient_checkpointing'):
                                trainer.model.module.gradient_checkpointing = value
                    
                    # Add property to DDP wrapper
                    trainer.model.__class__.gradient_checkpointing = property(
                        get_gradient_checkpointing,
                        set_gradient_checkpointing
                    )
                    logger.info_rank0("✅ Added gradient_checkpointing property to DDP model")
                
                # Forward LoRA adapter methods (critical for GRPO training)
                # Use accelerator.unwrap_model to properly access the original model
                if hasattr(original_model, 'disable_adapter'):
                    def disable_adapter_wrapper(*args, **kwargs):
                        logger.info_rank0("🔧 DDP disable_adapter called, unwrapping model with accelerator")
                        try:
                            # Method 1: Use accelerator.unwrap_model (recommended by unsloth-zoo)
                            unwrapped_model = trainer.accelerator.unwrap_model(trainer.model, keep_fp32_wrapper=False)
                            logger.info_rank0(f"🔍 Unwrapped model type: {type(unwrapped_model)}")
                            logger.info_rank0(f"🔍 Unwrapped model has disable_adapter: {hasattr(unwrapped_model, 'disable_adapter')}")
                            
                            if hasattr(unwrapped_model, 'disable_adapter'):
                                result = unwrapped_model.disable_adapter(*args, **kwargs)
                                logger.info_rank0("✅ Successfully called disable_adapter via unwrap_model")
                                return result
                            else:
                                logger.warning_rank0("⚠️ Unwrapped model doesn't have disable_adapter method")
                                # Fallback to direct .module access
                                logger.info_rank0("🔄 Falling back to direct .module access")
                                return trainer.model.module.disable_adapter(*args, **kwargs)
                        except Exception as e:
                            logger.warning_rank0(f"⚠️ accelerator.unwrap_model failed: {e}")
                            # Fallback to direct .module access
                            logger.info_rank0("🔄 Falling back to direct .module access")
                            return trainer.model.module.disable_adapter(*args, **kwargs)
                    
                    # Add method to DDP wrapper
                    trainer.model.disable_adapter = disable_adapter_wrapper
                    logger.info_rank0("✅ Added disable_adapter to DDP model")
                
                if hasattr(original_model, 'enable_adapter'):
                    def enable_adapter_wrapper(*args, **kwargs):
                        logger.info_rank0("🔧 DDP enable_adapter called, unwrapping model with accelerator")
                        try:
                            # Method 1: Use accelerator.unwrap_model (recommended by unsloth-zoo)
                            unwrapped_model = trainer.accelerator.unwrap_model(trainer.model, keep_fp32_wrapper=False)
                            logger.info_rank0(f"🔍 Unwrapped model type: {type(unwrapped_model)}")
                            logger.info_rank0(f"🔍 Unwrapped model has enable_adapter: {hasattr(unwrapped_model, 'enable_adapter')}")
                            
                            if hasattr(unwrapped_model, 'enable_adapter'):
                                result = unwrapped_model.enable_adapter(*args, **kwargs)
                                logger.info_rank0("✅ Successfully called enable_adapter via unwrap_model")
                                return result
                            else:
                                logger.warning_rank0("⚠️ Unwrapped model doesn't have enable_adapter method")
                                # Fallback to direct .module access
                                logger.info_rank0("🔄 Falling back to direct .module access")
                                return trainer.model.module.enable_adapter(*args, **kwargs)
                        except Exception as e:
                            logger.warning_rank0(f"⚠️ accelerator.unwrap_model failed: {e}")
                            # Fallback to direct .module access
                            logger.info_rank0("🔄 Falling back to direct .module access")
                            return trainer.model.module.enable_adapter(*args, **kwargs)
                    
                    # Add method to DDP wrapper
                    trainer.model.enable_adapter = enable_adapter_wrapper
                    logger.info_rank0("✅ Added enable_adapter to DDP model")
                
                # Forward other common LoRA/PEFT methods that might be needed
                lora_methods = ['set_adapter', 'get_adapter', 'active_adapter', 'merge_adapter', 'unmerge_adapter']
                for method_name in lora_methods:
                    if hasattr(original_model, method_name):
                        # Use default parameter to capture the method_name in the closure correctly
                        def create_forwarding_method(method_name=method_name):
                            def forwarding_method(*args, **kwargs):
                                logger.info_rank0(f"🔧 DDP {method_name} called, unwrapping model with accelerator")
                                try:
                                    # Method 1: Use accelerator.unwrap_model (recommended by unsloth-zoo)
                                    unwrapped_model = trainer.accelerator.unwrap_model(trainer.model, keep_fp32_wrapper=False)
                                    logger.info_rank0(f"🔍 Unwrapped model type: {type(unwrapped_model)}")
                                    logger.info_rank0(f"🔍 Unwrapped model has {method_name}: {hasattr(unwrapped_model, method_name)}")
                                    
                                    if hasattr(unwrapped_model, method_name):
                                        result = getattr(unwrapped_model, method_name)(*args, **kwargs)
                                        logger.info_rank0(f"✅ Successfully called {method_name} via unwrap_model")
                                        return result
                                    else:
                                        logger.warning_rank0(f"⚠️ Unwrapped model doesn't have {method_name} method")
                                        # Fallback to direct .module access
                                        logger.info_rank0("🔄 Falling back to direct .module access")
                                        return getattr(trainer.model.module, method_name)(*args, **kwargs)
                                except Exception as e:
                                    logger.warning_rank0(f"⚠️ accelerator.unwrap_model failed: {e}")
                                    # Fallback to direct .module access
                                    logger.info_rank0("🔄 Falling back to direct .module access")
                                    return getattr(trainer.model.module, method_name)(*args, **kwargs)
                            return forwarding_method
                        
                        setattr(trainer.model, method_name, create_forwarding_method())
                        logger.info_rank0(f"✅ Added {method_name} to DDP model")
                
                logger.info_rank0("✅ DDP gradient checkpointing and LoRA adapter compatibility setup completed")
            else:
                logger.warning_rank0("⚠️ Model is not DDP wrapped, skipping gradient checkpointing compatibility")
                
        except Exception as e:
            logger.error_rank0(f"❌ Failed to setup DDP gradient checkpointing compatibility: {e}")
            logger.warning_rank0("⚠️ Training may fail if gradient checkpointing is enabled")
            import traceback
            logger.error_rank0(f"   Exception traceback: {traceback.format_exc()}")
            # Don't raise - allow training to continue
    
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