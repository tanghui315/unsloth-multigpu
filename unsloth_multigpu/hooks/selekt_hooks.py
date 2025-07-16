"""
SeleKT integration hooks for Unsloth Multi-GPU
Provides seamless integration of SeleKT algorithm with existing Hook system
"""

import os
from typing import Any, Dict, Optional

from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from ..algorithms.selekt import SeleKTCallback, SeleKTConfig, SeleKTProcessor
from ..utils.ddp_logging import get_ddp_logger

logger = get_ddp_logger(__name__)


class SeleKTTrainerCallback(TrainerCallback):
    """
    HuggingFace Trainer compatible SeleKT callback
    Integrates seamlessly with existing training workflows
    """
    
    def __init__(self, config: SeleKTConfig):
        self.selekt_callback = SeleKTCallback(config)
        self.config = config
        
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs
    ):
        """Apply SeleKT when model is saved"""
        if model is not None:
            logger.info_rank0(f"Applying SeleKT to LoRA parameters at step {state.global_step}")
            
            # Apply SeleKT to LoRA parameters with step information
            self.selekt_callback.on_save(
                model=model,
                save_path=f"{args.output_dir}/checkpoint-{state.global_step}",
                current_step=state.global_step
            )
            
        return control
    
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs
    ):
        """Check for step-based SeleKT application"""
        if model is not None and self.config.step_frequency is not None:
            # Apply SeleKT based on step frequency
            self.selekt_callback.on_step(
                model=model,
                current_step=state.global_step
            )
            
        return control

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs
    ):
        """Apply SeleKT at the end of epoch"""
        if model is not None and self.config.epoch_frequency is not None:
            # Apply SeleKT based on epoch frequency
            self.selekt_callback.on_epoch_end(
                model=model,
                current_epoch=state.epoch,
                current_step=state.global_step
            )
            
        return control


class SeleKTHooks:
    """
    SeleKT hooks manager for integration with Unsloth Multi-GPU Hook system
    """
    
    def __init__(self):
        self.selekt_config: Optional[SeleKTConfig] = None
        self.hooks_applied = False
        self.original_functions = {}
        
    def enable_selekt(
        self,
        alpha: float = 0.05,
        apply_on_save: bool = True,
        save_selekt_checkpoints: bool = True,
        apply_frequency: int = 1,
        step_frequency: Optional[int] = None,
        max_interval_steps: Optional[int] = None,
        epoch_frequency: Optional[int] = None
    ) -> bool:
        """
        Enable SeleKT algorithm integration for LoRA parameters
        
        Args:
            alpha: SeleKT parameter selection ratio (0.05 = keep top 5%)
            apply_on_save: Whether to apply SeleKT when saving checkpoints
            save_selekt_checkpoints: Whether to save SeleKT-processed checkpoints
            apply_frequency: Apply SeleKT every N saves
            step_frequency: Apply SeleKT every N steps (independent of saves)
            max_interval_steps: Maximum steps between SeleKT applications
            epoch_frequency: Apply SeleKT every N epochs (recommended approach)
            
        Returns:
            bool: Whether SeleKT was successfully enabled
        """
        try:
            self.selekt_config = SeleKTConfig(
                alpha=alpha,
                enabled=True,
                apply_on_save=apply_on_save,
                save_selekt_checkpoints=save_selekt_checkpoints,
                apply_frequency=apply_frequency,
                step_frequency=step_frequency,
                max_interval_steps=max_interval_steps,
                epoch_frequency=epoch_frequency
            )
            
            config_info = f"α={alpha}"
            if step_frequency:
                config_info += f", step_freq={step_frequency}"
            if max_interval_steps:
                config_info += f", max_interval={max_interval_steps}"
                
            logger.info_rank0(f"✅ SeleKT enabled for LoRA parameters: {config_info}")
            return True
            
        except Exception as e:
            logger.error_rank0(f"❌ Failed to enable SeleKT: {e}")
            return False
    
    def apply_hooks(self) -> bool:
        """Apply SeleKT hooks to training system"""
        if not self.selekt_config or not self.selekt_config.enabled:
            logger.warning_rank0("SeleKT not configured, skipping hook application")
            return True
            
        try:
            # Hook into trainer creation to add SeleKT callback
            self._hook_trainer_creation()
            
            self.hooks_applied = True
            logger.info_rank0("✅ SeleKT hooks applied successfully")
            return True
            
        except Exception as e:
            logger.error_rank0(f"❌ Failed to apply SeleKT hooks: {e}")
            return False
    
    def remove_hooks(self) -> bool:
        """Remove SeleKT hooks and restore original functions"""
        if not self.hooks_applied:
            return True
            
        try:
            # Restore original functions
            for func_name, func_info in self.original_functions.items():
                setattr(func_info['module'], func_info['attr'], func_info['original'])
                
            self.hooks_applied = False
            self.original_functions.clear()
            logger.info_rank0("✅ SeleKT hooks removed successfully")
            return True
            
        except Exception as e:
            logger.error_rank0(f"❌ Failed to remove SeleKT hooks: {e}")
            return False
    
    def _hook_trainer_creation(self):
        """Hook trainer creation to automatically add SeleKT callback"""
        try:
            from trl import SFTTrainer

            # Save original __init__
            original_init = SFTTrainer.__init__
            self.original_functions['sft_trainer_init'] = {
                'module': SFTTrainer,
                'attr': '__init__',
                'original': original_init
            }
            
            def patched_init(self, *args, **kwargs):
                # Call original init
                result = original_init(self, *args, **kwargs)
                
                # Add SeleKT callback if configured
                if hasattr(self, 'callback_handler') and selekt_hooks.selekt_config:
                    selekt_callback = SeleKTTrainerCallback(selekt_hooks.selekt_config)
                    self.add_callback(selekt_callback)
                    logger.info_rank0("🔗 SeleKT callback added to trainer")
                
                return result
            
            # Replace function
            SFTTrainer.__init__ = patched_init
            logger.info_rank0("✅ SeleKT trainer hook applied")
            
        except ImportError:
            logger.warning_rank0("⚠️ TRL not found, SeleKT trainer hook skipped")
        except Exception as e:
            logger.error_rank0(f"❌ Failed to hook trainer creation: {e}")
            raise
    
    def get_hook_status(self) -> Dict[str, Any]:
        """Get SeleKT hooks status"""
        return {
            'selekt_enabled': self.selekt_config is not None and self.selekt_config.enabled,
            'hooks_applied': self.hooks_applied,
            'config': self.selekt_config.__dict__ if self.selekt_config else None,
            'active_hooks': list(self.original_functions.keys())
        }


# Global SeleKT hooks manager
selekt_hooks = SeleKTHooks()


def enable_selekt(
    alpha: float = 0.05,
    apply_on_save: bool = True,
    save_selekt_checkpoints: bool = True,
    apply_frequency: int = 1,
    step_frequency: Optional[int] = None,
    max_interval_steps: Optional[int] = None,
    epoch_frequency: Optional[int] = None
) -> bool:
    """
    Global function to enable SeleKT algorithm for LoRA parameters
    
    Usage:
        import unsloth_multigpu as ump
        
        # Enable SeleKT with 5% parameter selection
        ump.enable_selekt(alpha=0.05)
        
        # Enable multi-GPU training
        ump.enable_multi_gpu(num_gpus=2)
        
        # Train normally - SeleKT will be applied automatically to LoRA parameters
        trainer.train()
    """
    return selekt_hooks.enable_selekt(
        alpha=alpha,
        apply_on_save=apply_on_save,
        save_selekt_checkpoints=save_selekt_checkpoints,
        apply_frequency=apply_frequency,
        step_frequency=step_frequency,
        max_interval_steps=max_interval_steps,
        epoch_frequency=epoch_frequency
    )


def disable_selekt():
    """Disable SeleKT algorithm"""
    return selekt_hooks.remove_hooks()


def get_selekt_status() -> Dict[str, Any]:
    """Get SeleKT status information"""
    return selekt_hooks.get_hook_status()