"""
Trainer related Hook implementation
Responsible for handling Trainer configuration, callback functions, and performance monitoring, etc.
"""

import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


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
        Hook Trainer.train, make it call self-developed MultiGPUTrainer
        """
        try:
            from transformers import Trainer

            import unsloth_multigpu_prototype as uns_mgpu
            from unsloth_multigpu_prototype.core import MultiGPUTrainer

            # Save original train method
            original_train = Trainer.train
            self.original_functions['train'] = {
                'module': Trainer,
                'attr': 'train',
                'original': original_train
            }

            def patched_train(self, *args, **kwargs):
                """Patched train method, supports multi-GPU"""
                # If multi-GPU is not enabled, fall back to original method
                if not uns_mgpu.is_multi_gpu_enabled():
                    return original_train(self, *args, **kwargs)

                # If MultiGPUTrainer is not created, initialize
                if not hasattr(self, '_unsloth_multi_gpu_trainer'):
                    cfg = uns_mgpu.get_active_config()
                    self._unsloth_multi_gpu_trainer = MultiGPUTrainer(self, cfg)

                # Call self-developed trainer
                training_result = self._unsloth_multi_gpu_trainer.train()
                return training_result

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
    
    def _setup_multi_gpu_config(self):
        """Set multi-GPU configuration for Trainer"""
        # Add multi-GPU marker
        self._unsloth_multi_gpu_enabled = True
        self._unsloth_num_gpus = int(os.environ.get("UNSLOTH_MULTIGPU_NUM_GPUS", "1"))
        
        # Configure multi-GPU related attributes
        self._unsloth_performance_stats = {
            'multi_gpu_steps': 0,
            'total_multi_gpu_time': 0.0,
            'avg_step_time': 0.0
        }
        
        logger.info(f"🔧 Trainer multi-GPU configuration completed: {self._unsloth_num_gpus} GPUs")
    
    def _should_use_multi_gpu(self) -> bool:
        """Check if multi-GPU should be used"""
        return (
            os.environ.get("UNSLOTH_MULTIGPU_ENABLED") == "1" and
            hasattr(self, '_unsloth_multi_gpu_enabled') and
            self._unsloth_multi_gpu_enabled and
            self._unsloth_num_gpus > 1
        )
    
    def _update_performance_stats(self, duration: float, success: bool):
        """Update performance statistics"""
        self.performance_stats['hook_overhead_ms'] += duration * 1000
        self.performance_stats['total_hook_calls'] += 1
        
        if success:
            self.performance_stats['successful_hooks'] += 1
        else:
            self.performance_stats['failed_hooks'] += 1
    
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
            'trainer_hooks': ['trainer_init', 'train'],
            'performance_stats': self.performance_stats
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics information"""
        stats = self.performance_stats.copy()
        
        # Calculate average
        if stats['total_hook_calls'] > 0:
            stats['avg_hook_overhead_ms'] = stats['hook_overhead_ms'] / stats['total_hook_calls']
            stats['success_rate'] = stats['successful_hooks'] / stats['total_hook_calls']
        else:
            stats['avg_hook_overhead_ms'] = 0.0
            stats['success_rate'] = 0.0
        
        return stats
    
    def reset_performance_stats(self):
        """Reset performance statistics"""
        self.performance_stats = {
            'hook_overhead_ms': 0.0,
            'total_hook_calls': 0,
            'successful_hooks': 0,
            'failed_hooks': 0
        }
        logger.info("🔄 Performance statistics reset")
    
    def __del__(self):
        """Destructor, ensure Hooks are correctly removed"""
        if self.hooks_applied:
            self.remove_hooks() 