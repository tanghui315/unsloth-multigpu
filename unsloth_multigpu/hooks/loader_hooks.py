"""
Model loading related Hook implementation
Responsible for replacing model loading related functions such as FastLanguageModel.from_pretrained
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class LoaderHooks:
    """
    Model loading related Hook manager
    
    Responsible:
    1. Dynamic replacement of FastLanguageModel.from_pretrained function
    2. Multi-GPU model loading configuration and marking
    3. Error handling during model loading process
    4. Integration configuration with multi-GPU training
    """
    
    def __init__(self):
        """Initialize model loading Hook manager"""
        self.original_functions = {}
        self.hooks_applied = False
        
        logger.info("🔧 Initialize LoaderHooks manager")
    
    def apply_hooks(self) -> bool:
        """
        Apply all model loading related Hooks
        
        Returns:
            bool: Whether Hooks are successfully applied
        """
        if self.hooks_applied:
            logger.warning("Hooks already applied, skipping repeated application")
            return True
        
        success = True
        
        # Apply model loading Hooks
        success &= self._hook_model_loader()
        
        if success:
            self.hooks_applied = True
            logger.info("✅ All loading Hooks applied successfully")
        else:
            logger.error("❌ Some loading Hooks failed to apply")
        
        return success
    
    def remove_hooks(self) -> bool:
        """
        Remove all loading Hooks, restore original functions
        
        Returns:
            bool: Whether Hooks are successfully removed
        """
        if not self.hooks_applied:
            logger.warning("Loading Hooks not applied, no need to remove")
            return True
        
        success = True
        
        for func_name, func_info in self.original_functions.items():
            try:
                setattr(func_info['module'], func_info['attr'], func_info['original'])
                logger.info(f"✅ Restored function: {func_name}")
            except Exception as e:
                logger.error(f"❌ Restoring function {func_name} failed: {e}")
                success = False
        
        if success:
            self.hooks_applied = False
            self.original_functions.clear()
            logger.info("✅ All loading Hooks removed successfully")
        
        return success
    
    def _hook_model_loader(self) -> bool:
        """
        Hook FastLanguageModel.from_pretrained function
        
        Returns:
            bool: Whether Hook is successfully applied
        """
        try:
            # Try importing unsloth
            import unsloth
            from unsloth.models.loader import FastLanguageModel

            # Save original function
            original_from_pretrained = FastLanguageModel.from_pretrained
            self.original_functions['from_pretrained'] = {
                'module': FastLanguageModel,
                'attr': 'from_pretrained',
                'original': original_from_pretrained
            }
            
            @classmethod
            def patched_from_pretrained(cls, *args, **kwargs):
                """Support multi-GPU model loading"""
                return self._execute_multi_gpu_loading(
                    original_from_pretrained, cls, *args, **kwargs
                )
            
            # Replace function
            FastLanguageModel.from_pretrained = patched_from_pretrained
            logger.info("✅ Hook FastLanguageModel.from_pretrained applied successfully")
            return True
            
        except ImportError:
            logger.warning("⚠️ unsloth not found, skipping model loading Hook")
            return True  # Not a fatal error
        except Exception as e:
            logger.error(f"❌ Hook model loading failed: {e}")
            return False
    
    def _execute_multi_gpu_loading(self, original_func, cls, *args, **kwargs) -> Tuple[Any, Any]:
        """
        Execute core logic of multi-GPU model loading
        
        Args:
            original_func: Original loading function
            cls: Class object
            *args: Position arguments
            **kwargs: Keyword arguments
            
        Returns:
            Tuple: (model, tokenizer)
        """
        # Extract multi-GPU related parameters
        enable_multi_gpu = kwargs.pop('enable_multi_gpu', 
                                     os.environ.get("UNSLOTH_MULTIGPU_ENABLED") == "1")
        num_gpus = kwargs.pop('num_gpus', 
                             int(os.environ.get("UNSLOTH_MULTIGPU_NUM_GPUS", "1")))
        
        # Multi-GPU specific configuration
        multi_gpu_config = kwargs.pop('multi_gpu_config', {})
        
        logger.info(f"🔄 Loading model: enable_multi_gpu={enable_multi_gpu}, num_gpus={num_gpus}")
        
        # 🔧 DDP environment detection and device setup
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            import torch
            rank = int(os.environ.get('RANK', 0))
            local_rank = int(os.environ.get('LOCAL_RANK', rank))
            world_size = int(os.environ.get('WORLD_SIZE', 1))
            
            logger.info(f"🚀 Detected DDP environment: rank={rank}, local_rank={local_rank}, world_size={world_size}")
            
            # Set correct CUDA device before model loading
            if torch.cuda.is_available() and local_rank < torch.cuda.device_count():
                torch.cuda.set_device(local_rank)
                logger.info(f"✅ Set CUDA device: GPU {local_rank}")
                
                # Ensure model loads to the correct device
                if 'device_map' not in kwargs:
                    kwargs['device_map'] = f'cuda:{local_rank}'
                    logger.info(f"🎯 Force model loading to cuda:{local_rank}")
                
                # Clear CUDA memory cache
                torch.cuda.empty_cache()
            else:
                logger.warning(f"⚠️ Unable to set CUDA device {local_rank}")
        
        try:
            # Call original loading function
            model, tokenizer = original_func(*args, **kwargs)
            
            # GPU memory usage monitoring
            if 'RANK' in os.environ and torch.cuda.is_available():
                import torch
                rank = int(os.environ.get('RANK', 0))
                local_rank = int(os.environ.get('LOCAL_RANK', rank))
                allocated = torch.cuda.memory_allocated(local_rank) / 1024**3
                cached = torch.cuda.memory_reserved(local_rank) / 1024**3
                logger.info(f"📊 Rank {rank} GPU {local_rank} memory after model loading: {allocated:.2f}GB (allocated), {cached:.2f}GB (cached)")
            
            # If multi-GPU is enabled, configure model
            if enable_multi_gpu and num_gpus > 1:
                model = self._configure_model_for_multi_gpu(
                    model, num_gpus, multi_gpu_config
                )
                
                logger.info(f"🔧 Model configured for multi-GPU training: {num_gpus} GPUs")
                
                # Check core component availability
                try:
                    from ..core import MultiGPUManager
                    logger.info(f"✅ Core components ready, supporting multi-GPU training")
                except ImportError:
                    logger.warning(f"⚠️ Core components not available, using fallback mode")
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"❌ Multi-GPU model loading failed: {e}")
            logger.error("📋 Error details:", exc_info=True)
            
            # Fallback to original loading
            logger.info("🔄 Falling back to original model loading...")
            return original_func(*args, **kwargs)
    
    def _configure_model_for_multi_gpu(self, model, num_gpus: int, config: Dict[str, Any]):
        """
        Configure model for multi-GPU training
        
        Args:
            model: Loaded model
            num_gpus: Number of GPUs
            config: Multi-GPU configuration
            
        Returns:
            Configured model
        """
        # Mark model as supporting multi-GPU
        model._unsloth_multi_gpu_enabled = True
        model._unsloth_target_num_gpus = num_gpus
        model._unsloth_multi_gpu_config = config
        
        # Set model specific configuration
        if 'gradient_checkpointing' in config:
            if hasattr(model, 'gradient_checkpointing_enable'):
                if config['gradient_checkpointing']:
                    model.gradient_checkpointing_enable()
                    logger.info("✅ Enabled gradient checkpointing")
                else:
                    model.gradient_checkpointing_disable()
                    logger.info("📝 Disabled gradient checkpointing")
        
        # Set mixed precision configuration
        if 'mixed_precision' in config:
            model._unsloth_mixed_precision = config['mixed_precision']
            logger.info(f"🎯 Mixed precision setting: {config['mixed_precision']}")
        
        # Set memory optimization configuration
        if 'memory_optimization' in config:
            model._unsloth_memory_optimization = config['memory_optimization']
            logger.info(f"💾 Memory optimization setting: {config['memory_optimization']}")
        
        # Set batch slicing strategy
        if 'batch_sharding_strategy' in config:
            model._unsloth_batch_strategy = config['batch_sharding_strategy']
            logger.info(f"📊 Batch slicing strategy: {config['batch_sharding_strategy']}")
        
        # Set gradient aggregation method
        if 'gradient_aggregation' in config:
            model._unsloth_grad_aggregation = config['gradient_aggregation']
            logger.info(f"🔗 Gradient aggregation method: {config['gradient_aggregation']}")
        
        logger.info(f"🏷️ Model multi-GPU marking completed")
        
        return model
    
    def _validate_multi_gpu_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate the validity of multi-GPU configuration
        
        Args:
            config: Multi-GPU configuration dictionary
            
        Returns:
            bool: Whether configuration is valid
        """
        valid_keys = {
            'gradient_checkpointing', 'mixed_precision', 'memory_optimization',
            'batch_sharding_strategy', 'gradient_aggregation', 'device_placement'
        }
        
        # Check for invalid configuration keys
        invalid_keys = set(config.keys()) - valid_keys
        if invalid_keys:
            logger.warning(f"⚠️ Invalid multi-GPU configuration keys: {invalid_keys}")
        
        # Validate specific configuration values
        if 'gradient_aggregation' in config:
            valid_methods = ['mean', 'sum', 'weighted_mean', 'median']
            if config['gradient_aggregation'] not in valid_methods:
                logger.error(f"❌ Invalid gradient aggregation method: {config['gradient_aggregation']}")
                return False
        
        if 'batch_sharding_strategy' in config:
            valid_strategies = ['uniform', 'adaptive', 'weighted']
            if config['batch_sharding_strategy'] not in valid_strategies:
                logger.error(f"❌ Invalid batch slicing strategy: {config['batch_sharding_strategy']}")
                return False
        
        return True
    
    def get_hook_status(self) -> Dict[str, Any]:
        """
        Get Hook status information
        
        Returns:
            Dict: Hook status statistics
        """
        return {
            'hooks_applied': self.hooks_applied,
            'active_hooks': list(self.original_functions.keys()),
            'num_hooks': len(self.original_functions),
            'loader_hooks': ['from_pretrained']
        }
    
    def get_model_multi_gpu_info(self, model) -> Optional[Dict[str, Any]]:
        """
        Get model multi-GPU configuration information
        
        Args:
            model: Model object
            
        Returns:
            Dict: Multi-GPU configuration info, or None if not supported
        """
        if not hasattr(model, '_unsloth_multi_gpu_enabled'):
            return None
        
        return {
            'enabled': getattr(model, '_unsloth_multi_gpu_enabled', False),
            'num_gpus': getattr(model, '_unsloth_target_num_gpus', 1),
            'config': getattr(model, '_unsloth_multi_gpu_config', {}),
            'mixed_precision': getattr(model, '_unsloth_mixed_precision', None),
            'memory_optimization': getattr(model, '_unsloth_memory_optimization', None),
            'batch_strategy': getattr(model, '_unsloth_batch_strategy', None),
            'grad_aggregation': getattr(model, '_unsloth_grad_aggregation', None)
        }
    
    def __del__(self):
        """Destructor, ensure Hooks are properly removed"""
        if self.hooks_applied:
            self.remove_hooks() 