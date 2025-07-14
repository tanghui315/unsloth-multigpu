"""
Unsloth Multi-GPU external extension package
Multi-GPU support via dynamic function replacement, fully compatible with the open-source version
"""

import logging
import os
import sys
import warnings
from typing import Any, Dict, Optional

# Import DDP components
try:
    from .core import DDPManager, DDPTrainer, launch_ddp_training
    from .utils import MultiGPUConfig
    CORE_COMPONENTS_AVAILABLE = True
except ImportError as e:
    CORE_COMPONENTS_AVAILABLE = False
    logging.warning(f"Failed to import DDP components: {e}")

# Import hook components
try:
    from .hooks import LoaderHooks, TrainerHooks, TrainingHooks
    HOOKS_AVAILABLE = True
except ImportError as e:
    HOOKS_AVAILABLE = False
    logging.warning(f"Failed to import hook components: {e}")

__version__ = "1.0.0"
__all__ = ["enable_multi_gpu", "disable_multi_gpu", "is_multi_gpu_enabled", "get_multi_gpu_status", "get_active_config"]

# Global state flags
_MULTIGPU_ENABLED = False
_HOOK_MANAGERS = {}
_ACTIVE_CONFIG: Optional[MultiGPUConfig] = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def enable_multi_gpu(
    num_gpus: Optional[int] = None,
    batch_size_per_gpu: int = 2,
    enable_memory_optimization: bool = True,
    enable_gradient_checkpointing: bool = True,
    debug: bool = False,
    # DDP options
    ddp_backend: str = "nccl",  # nccl for GPU, gloo for CPU
    find_unused_parameters: bool = False
):
    """
    Enable Unsloth Multi-GPU support using PyTorch DDP
    
    Args:
        num_gpus: Number of GPUs to use, None for auto-detect
        batch_size_per_gpu: Batch size per GPU device
        enable_memory_optimization: Whether to enable memory optimization
        enable_gradient_checkpointing: Whether to enable gradient checkpointing
        debug: Whether to enable debug mode
        ddp_backend: DDP backend ("nccl" for GPU, "gloo" for CPU)
        find_unused_parameters: DDP find_unused_parameters setting
    """
    global _MULTIGPU_ENABLED, _HOOK_MANAGERS, _ACTIVE_CONFIG
    
    if _MULTIGPU_ENABLED:
        warnings.warn("Multi-GPU support is already enabled.")
        return

    # Check component availability
    if not CORE_COMPONENTS_AVAILABLE:
        logger.error("❌ Core components not available, cannot enable Multi-GPU support")
        raise RuntimeError("Failed to import core components, please check installation")
    
    if not HOOKS_AVAILABLE:
        logger.error("❌ Hook components not available, cannot enable Multi-GPU support")
        raise RuntimeError("Failed to import hook components, please check installation")

    try:
        # Validate environment
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Multi-GPU requires CUDA.")
        
        available_gpus = torch.cuda.device_count()
        if available_gpus < 2:
            logger.warning(f"Only {available_gpus} GPU(s) available. Continuing in single GPU mode.")
            return
        
        if num_gpus is None:
            num_gpus = available_gpus
        elif num_gpus > available_gpus:
            raise ValueError(f"Requested {num_gpus} GPUs, but only {available_gpus} available.")

        # Create DDP config
        if CORE_COMPONENTS_AVAILABLE:
            _ACTIVE_CONFIG = MultiGPUConfig(
                num_gpus=num_gpus,
                batch_size_per_gpu=batch_size_per_gpu,
                memory_optimization=enable_memory_optimization,
                enable_gradient_checkpointing=enable_gradient_checkpointing,
                log_level="DEBUG" if debug else "INFO",
                # DDP settings
                ddp_backend=ddp_backend,
                find_unused_parameters=find_unused_parameters
            )
        else:
            # Fallback: only store basic parameters
            _ACTIVE_CONFIG = {
                'num_gpus': num_gpus,
                'batch_size_per_gpu': batch_size_per_gpu,
                'memory_optimization': enable_memory_optimization,
                'enable_gradient_checkpointing': enable_gradient_checkpointing,
                'debug': debug
            }

        # Set environment variables (for backward compatibility)
        os.environ["UNSLOTH_MULTIGPU_ENABLED"] = "1"
        os.environ["UNSLOTH_MULTIGPU_NUM_GPUS"] = str(num_gpus)
        os.environ["UNSLOTH_MULTIGPU_DEBUG"] = str(debug)
        
        # Initialize and apply hooks
        success = _initialize_hook_managers()
        if not success:
            raise RuntimeError("Failed to initialize hook managers")
        
        success = _apply_all_hooks()
        if not success:
            raise RuntimeError("Failed to apply hooks")
        
        _MULTIGPU_ENABLED = True
        
        print(f"🚀 Unsloth Multi-GPU enabled: {num_gpus} GPUs")
        print(f"📊 Batch size per GPU: {batch_size_per_gpu}")
        print(f"🔧 DDP backend: {ddp_backend}")
        
    except Exception as e:
        logger.error(f"Failed to enable Multi-GPU: {e}")
        # Clean up already applied hooks
        _cleanup_hook_managers()
        _ACTIVE_CONFIG = None
        raise

def disable_multi_gpu():
    """Disable Multi-GPU support and restore original functions"""
    global _MULTIGPU_ENABLED, _HOOK_MANAGERS, _ACTIVE_CONFIG
    
    if not _MULTIGPU_ENABLED:
        return
    
    # Remove all hooks
    _remove_all_hooks()
    
    # Clean up hook managers
    _cleanup_hook_managers()
    
    _MULTIGPU_ENABLED = False
    _ACTIVE_CONFIG = None
    
    # Clean up environment variables
    for key in ["UNSLOTH_MULTIGPU_ENABLED", "UNSLOTH_MULTIGPU_NUM_GPUS", "UNSLOTH_MULTIGPU_DEBUG"]:
        os.environ.pop(key, None)
    
    print("🔄 Unsloth Multi-GPU disabled, restored to single GPU mode")

def is_multi_gpu_enabled() -> bool:
    """Check if Multi-GPU is enabled"""
    return _MULTIGPU_ENABLED

def get_active_config() -> Optional[MultiGPUConfig]:
    """Get the currently active MultiGPUConfig object"""
    return _ACTIVE_CONFIG

def get_multi_gpu_status() -> Dict[str, Any]:
    """
    Get detailed status of the Multi-GPU system
    
    Returns:
        Dict: Detailed system status information
    """
    status = {
        'enabled': _MULTIGPU_ENABLED,
        'core_components_available': CORE_COMPONENTS_AVAILABLE,
        'hooks_available': HOOKS_AVAILABLE,
        'active_config': _ACTIVE_CONFIG.to_dict() if _ACTIVE_CONFIG else None,
        'num_gpus': int(os.environ.get("UNSLOTH_MULTIGPU_NUM_GPUS", "1")),
        'debug_mode': os.environ.get("UNSLOTH_MULTIGPU_DEBUG", "False") == "True"
    }
    
    # Add hook status
    if _HOOK_MANAGERS:
        status['hook_managers'] = {}
        for name, manager in _HOOK_MANAGERS.items():
            if hasattr(manager, 'get_hook_status'):
                status['hook_managers'][name] = manager.get_hook_status()
    
    # Add GPU info
    try:
        import torch
        if torch.cuda.is_available():
            status['gpu_info'] = {
                'available_gpus': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            }
    except ImportError:
        status['gpu_info'] = {'error': 'PyTorch not available'}
    except Exception as e:
        status['gpu_info'] = {'error': f'Failed to get GPU info: {e}'}
    
    return status

def _initialize_hook_managers() -> bool:
    """
    Initialize all hook managers
    
    Returns:
        bool: Whether initialization succeeded
    """
    global _HOOK_MANAGERS
    
    try:
        # Initialize training hook manager
        _HOOK_MANAGERS['training'] = TrainingHooks()
        logger.info("✅ Training hook manager initialized")
        
        # Initialize loader hook manager
        _HOOK_MANAGERS['loader'] = LoaderHooks()
        logger.info("✅ Loader hook manager initialized")
        
        # Initialize trainer hook manager
        _HOOK_MANAGERS['trainer'] = TrainerHooks()
        logger.info("✅ Trainer hook manager initialized")
        
        logger.info(f"🔧 All hook managers initialized: {len(_HOOK_MANAGERS)} managers")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize hook managers: {e}")
        return False

def _apply_all_hooks() -> bool:
    """
    Apply all hooks
    
    Returns:
        bool: Whether all hooks were applied successfully
    """
    success = True
    
    for name, manager in _HOOK_MANAGERS.items():
        try:
            if hasattr(manager, 'apply_hooks'):
                manager_success = manager.apply_hooks()
                if manager_success:
                    logger.info(f"✅ {name} hook applied successfully")
                else:
                    logger.error(f"❌ {name} hook application failed")
                    success = False
        except Exception as e:
            logger.error(f"❌ Exception during {name} hook application: {e}")
            success = False
    
    if success:
        logger.info("🎯 All hooks applied")
    else:
        logger.error("⚠️ Some hooks failed to apply")
    
    return success

def _remove_all_hooks() -> bool:
    """
    Remove all hooks
    
    Returns:
        bool: Whether all hooks were removed successfully
    """
    success = True
    
    for name, manager in _HOOK_MANAGERS.items():
        try:
            if hasattr(manager, 'remove_hooks'):
                manager_success = manager.remove_hooks()
                if manager_success:
                    logger.info(f"✅ {name} hook removed successfully")
                else:
                    logger.error(f"❌ {name} hook removal failed")
                    success = False
        except Exception as e:
            logger.error(f"❌ Exception during {name} hook removal: {e}")
            success = False
    
    return success

def _cleanup_hook_managers():
    """Clean up hook managers"""
    global _HOOK_MANAGERS
    
    for name, manager in _HOOK_MANAGERS.items():
        try:
            if hasattr(manager, '__del__'):
                del manager
            logger.info(f"🧹 {name} hook manager cleaned up")
        except Exception as e:
            logger.warning(f"⚠️ Failed to clean up {name} hook manager: {e}")
    
    _HOOK_MANAGERS.clear()
    logger.info("🧹 All hook managers cleaned up")

# Legacy hook functions for backward compatibility
def _apply_hooks():
    """Legacy hook application function, kept for backward compatibility"""
    logger.warning("⚠️ Using legacy hook system, upgrade to modular hooks is recommended")
    return _apply_all_hooks()

def _restore_original_functions():
    """Legacy hook restore function, kept for backward compatibility"""
    logger.warning("⚠️ Using legacy hook restore, upgrade to modular hooks is recommended")
    return _remove_all_hooks()

# Check if should auto-enable
if os.environ.get("UNSLOTH_AUTO_ENABLE_MULTIGPU", "0") == "1":
    try:
        enable_multi_gpu()
    except Exception as e:
        logger.warning(f"Auto-enable failed: {e}")

# If this package is directly imported, try to auto-enable
if __name__ != "__main__":
    # Detect if in a suitable environment
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            logger.info("🔍 Detected multi-GPU environment, ready to enable when needed")
            if CORE_COMPONENTS_AVAILABLE:
                logger.info("✅ Core components loaded successfully")
            else:
                logger.warning("⚠️ Core components not available")
            
            if HOOKS_AVAILABLE:
                logger.info("✅ Modular hook system loaded successfully")
            else:
                logger.warning("⚠️ Modular hook system not available")
    except ImportError:
        pass 