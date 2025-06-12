"""
Training related Hook implementation
Responsible for replacing training related functions such as unsloth_train and get_max_steps
"""

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TrainingHooks:
    """
    Training related Hook manager
    
    Responsible:
    1. Dynamic replacement of unsloth_train function
    2. Removal of multi-GPU restriction from get_max_steps function
    3. Error handling and rollback mechanism for training process
    4. Collection of training statistics
    """
    
    def __init__(self):
        """Initialize TrainingHooks manager"""
        self.original_functions = {}
        self.hooks_applied = False
        
        logger.info("🔧 Initialize TrainingHooks manager")
    
    def apply_hooks(self) -> bool:
        """
        Apply all training related Hooks
        
        Returns:
            bool: Whether Hooks are successfully applied
        """
        if self.hooks_applied:
            logger.warning("Training Hooks already applied, skipping repeated application")
            return True
        
        success = True
        
        # Apply various Hooks
        success &= self._hook_get_max_steps()
        success &= self._hook_unsloth_train()
        
        if success:
            self.hooks_applied = True
            logger.info("✅ All training Hooks applied successfully")
        else:
            logger.error("❌ Some training Hooks failed to apply")
        
        return success
    
    def remove_hooks(self) -> bool:
        """
        Remove all training Hooks, restore original functions
        
        Returns:
            bool: Whether Hooks are successfully removed
        """
        if not self.hooks_applied:
            logger.warning("Training Hooks not applied, no need to remove")
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
            logger.info("✅ All training Hooks removed successfully")
        
        return success
    
    def _hook_get_max_steps(self) -> bool:
        """
        Hook get_max_steps function, remove multi-GPU check
        
        Returns:
            bool: Whether Hook is successfully applied
        """
        try:
            import unsloth_zoo.training_utils as training_utils

            # Save original function
            original_get_max_steps = training_utils.get_max_steps
            self.original_functions['get_max_steps'] = {
                'module': training_utils,
                'attr': 'get_max_steps',
                'original': original_get_max_steps
            }
            
            def patched_get_max_steps(training_args, n_training_samples, train_dataset):
                """Remove multi-GPU restriction from get_max_steps"""
                # Temporarily disable multi-GPU check
                original_world_size = training_args.world_size
                
                if os.environ.get("UNSLOTH_MULTIGPU_ENABLED") == "1":
                    # Temporarily set to 1 to pass check
                    training_args.world_size = 1
                
                try:
                    result = original_get_max_steps(training_args, n_training_samples, train_dataset)
                finally:
                    # Restore original world_size
                    training_args.world_size = original_world_size
                
                return result
            
            # Replace function
            training_utils.get_max_steps = patched_get_max_steps
            logger.info("✅ Hook get_max_steps applied successfully")
            return True
            
        except ImportError:
            logger.warning("⚠️ unsloth_zoo.training_utils not found, skipping get_max_steps Hook")
            return True  # Not fatal error
        except Exception as e:
            logger.error(f"❌ Hook get_max_steps failed: {e}")
            return False
    
    def _hook_unsloth_train(self) -> bool:
        """
        Hook unsloth_train function, add multi-GPU support
        
        Returns:
            bool: Whether Hook is successfully applied
        """
        try:
            import unsloth_zoo.training_utils as training_utils

            # Save original function
            original_unsloth_train = training_utils.unsloth_train
            self.original_functions['unsloth_train'] = {
                'module': training_utils,
                'attr': 'unsloth_train',
                'original': original_unsloth_train
            }
            
            def patched_unsloth_train(trainer):
                """Support multi-GPU unsloth_train"""
                # Check if multi-GPU is enabled
                if os.environ.get("UNSLOTH_MULTIGPU_ENABLED") != "1":
                    return original_unsloth_train(trainer)
                
                num_gpus = int(os.environ.get("UNSLOTH_MULTIGPU_NUM_GPUS", "1"))
                if num_gpus == 1:
                    return original_unsloth_train(trainer)
                
                # Check if core components are available
                try:
                    from ..core import MultiGPUTrainer
                    core_available = True
                except ImportError:
                    core_available = False
                
                if not core_available:
                    logger.warning("⚠️ Core components not available, falling back to single-GPU training")
                    return original_unsloth_train(trainer)
                
                logger.info(f"🚀 Starting multi-GPU training: {num_gpus} GPUs")
                
                # Use new multi-GPU training logic
                try:
                    result = self._execute_multi_gpu_training(trainer, num_gpus, original_unsloth_train)
                    return result
                except Exception as e:
                    logger.error(f"❌ Multi-GPU training failed, falling back to single-GPU: {e}")
                    logger.error("📋 Error details:", exc_info=True)
                    return original_unsloth_train(trainer)
            
            # Replace function
            training_utils.unsloth_train = patched_unsloth_train
            logger.info("✅ Hook unsloth_train applied successfully")
            return True
            
        except ImportError:
            logger.warning("⚠️ unsloth_zoo.training_utils not found, skipping unsloth_train Hook")
            return True  # Not fatal error
        except Exception as e:
            logger.error(f"❌ Hook unsloth_train failed: {e}")
            return False
    
    def _execute_multi_gpu_training(self, trainer, num_gpus: int, original_func) -> Dict[str, Any]:
        """
        Execute core logic of multi-GPU training
        
        Args:
            trainer: Original trainer object
            num_gpus: Number of GPUs
            original_func: Original training function
            
        Returns:
            Dict: Training result statistics
        """
        logger.info(f"🔄 Initializing multi-GPU training components...")
        
        # Collect all error information instead of immediately falling back
        errors = []
        
        # Import core components and configuration
        from .. import get_active_config
        from ..core import MultiGPUTrainer
        from ..utils import MultiGPUConfig

        # Get active configuration
        active_config = get_active_config()
        if not active_config:
            error_msg = "❌ No active multi-GPU configuration found. Please ensure enable_multi_gpu() function is called"
            errors.append(error_msg)
            logger.error(error_msg)
            return self._handle_fallback_with_details(trainer, original_func, errors)

        # Handle configuration type compatibility
        config_obj = None
        if isinstance(active_config, dict):
            try:
                config_obj = MultiGPUConfig(**active_config)
            except Exception as e:
                error_msg = f"❌ Configuration object conversion failed: {e}. Configuration content: {active_config}"
                errors.append(error_msg)
                logger.error(error_msg)
        elif isinstance(active_config, MultiGPUConfig):
            config_obj = active_config
        else:
            error_msg = f"❌ Unknown configuration type: {type(active_config)}. Expected MultiGPUConfig or dict"
            errors.append(error_msg)
            logger.error(error_msg)
        
        if config_obj is None:
            return self._handle_fallback_with_details(trainer, original_func, errors)
        
        # Check if there is training data
        if not hasattr(trainer, 'train_dataset') or not trainer.train_dataset:
            error_msg = "⚠️ No training data found. trainer.train_dataset is empty or does not exist"
            errors.append(error_msg)
            logger.warning(error_msg)
            return self._handle_fallback_with_details(trainer, original_func, errors)
        
        # Validate GPU environment
        gpu_errors = self._validate_gpu_environment(config_obj)
        if gpu_errors:
            errors.extend(gpu_errors)
            return self._handle_fallback_with_details(trainer, original_func, errors)
        
        try:
            # Use configuration object to create multi-GPU trainer
            logger.info(f"⚙️ Training with configuration: batch_size_per_gpu={config_obj.batch_size_per_gpu}, gradient_aggregation={config_obj.gradient_aggregation}")
            
            multi_gpu_trainer = MultiGPUTrainer(
                original_trainer=trainer,
                config=config_obj
            )
            
            # Set up trainer
            multi_gpu_trainer.setup()
            
            logger.info(f"✅ Multi-GPU trainer initialization completed")
            
            # Execute training
            return multi_gpu_trainer.train()
            
        except Exception as e:
            error_msg = f"❌ Multi-GPU trainer execution failed: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
            return self._handle_fallback_with_details(trainer, original_func, errors)
    
    def _validate_gpu_environment(self, config: 'MultiGPUConfig') -> List[str]:
        """Validate GPU environment, return error list"""
        import torch
        errors = []
        
        if not torch.cuda.is_available():
            errors.append("❌ CUDA not available. Please install PyTorch version with CUDA support")
            return errors
        
        available_gpus = torch.cuda.device_count()
        if available_gpus < config.num_gpus:
            errors.append(f"❌ Available GPU count {available_gpus} < Required count {config.num_gpus}")
        
        # Check each GPU's memory
        for gpu_id in range(min(available_gpus, config.num_gpus)):
            try:
                with torch.cuda.device(gpu_id):
                    props = torch.cuda.get_device_properties(gpu_id)
                    free_memory_gb = props.total_memory / 1024**3
                    
                    # Estimate minimum memory needed (very rough estimate)
                    estimated_min_memory = config.batch_size_per_gpu * 0.5  # GB per sample
                    
                    if free_memory_gb < estimated_min_memory:
                        errors.append(f"⚠️ GPU {gpu_id} Memory may be insufficient: {free_memory_gb:.1f}GB < Estimated need {estimated_min_memory:.1f}GB")
            except Exception as e:
                errors.append(f"❌ Unable to check GPU {gpu_id}: {e}")
        
        return errors
    
    def _handle_fallback_with_details(self, trainer, original_func, errors: List[str]) -> Dict[str, Any]:
        """Handle fallback, provide detailed error information"""
        
        # Generate detailed error report
        error_report = "\n" + "="*50 + "\n"
        error_report += "🚨 Multi-GPU training failed, falling back to single-GPU training\n"
        error_report += "="*50 + "\n"
        error_report += "Error details:\n"
        
        for i, error in enumerate(errors, 1):
            error_report += f"{i}. {error}\n"
        
        error_report += "\n💡 Solution suggestions:\n"
        error_report += "• Check if enable_multi_gpu() is called correctly\n"
        error_report += "• Verify GPU count and memory are sufficient\n"
        error_report += "• Try reducing batch_size_per_gpu parameter\n"
        error_report += "• View above detailed error information for targeted repair\n"
        error_report += "="*50
        
        logger.error(error_report)
        
        # If user set strict mode, raise exception instead of falling back
        if os.environ.get("UNSLOTH_MULTIGPU_STRICT", "0") == "1":
            raise RuntimeError(f"Multi-GPU training failed, error count: {len(errors)}. Details please check log.")
        
        logger.warning("🔄 Executing single-GPU training...")
        return original_func(trainer)
    
    def _extract_optimizer_config(self, trainer) -> Dict[str, Any]:
        """
        Extract optimizer configuration from original trainer
        
        Args:
            trainer: Trainer object
            
        Returns:
            Dict: Optimizer configuration
        """
        if hasattr(trainer, 'optimizer') and trainer.optimizer:
            optimizer = trainer.optimizer
            return {
                'class': optimizer.__class__,
                'kwargs': {
                    'lr': optimizer.param_groups[0].get('lr', 2e-5),
                    'weight_decay': optimizer.param_groups[0].get('weight_decay', 0.01),
                    'eps': optimizer.param_groups[0].get('eps', 1e-8),
                }
            }
        else:
            # Default configuration
            import torch.optim as optim
            return {
                'class': optim.AdamW,
                'kwargs': {
                    'lr': 2e-5,
                    'weight_decay': 0.01,
                    'eps': 1e-8,
                }
            }
    
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
            'training_hooks': ['get_max_steps', 'unsloth_train']
        }
    
    def __del__(self):
        """Destructor function, ensure Hooks are correctly removed"""
        if self.hooks_applied:
            self.remove_hooks() 