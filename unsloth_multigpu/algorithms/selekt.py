"""
SeleKT Algorithm Implementation for Unsloth Multi-GPU
Sparse parameter selection algorithm for efficient LoRA training
"""

import gc
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from tqdm import tqdm

from ..utils.ddp_logging import get_ddp_logger

logger = get_ddp_logger(__name__)


class SeleKTConfig:
    """Configuration for SeleKT algorithm"""
    
    def __init__(
        self,
        alpha: float = 0.05,
        enabled: bool = True,
        save_selekt_checkpoints: bool = True,
        apply_on_save: bool = True,
        apply_frequency: int = 1,  # Apply every N saves
        memory_efficient: bool = True,
        step_frequency: Optional[int] = None,  # Apply every N steps (independent of saves)
        max_interval_steps: Optional[int] = None,  # Maximum steps between applications
        epoch_frequency: Optional[int] = None,  # Apply every N epochs (recommended)
    ):
        self.alpha = alpha
        self.enabled = enabled
        self.save_selekt_checkpoints = save_selekt_checkpoints
        self.apply_on_save = apply_on_save
        self.apply_frequency = apply_frequency
        self.memory_efficient = memory_efficient
        self.step_frequency = step_frequency
        self.max_interval_steps = max_interval_steps
        self.epoch_frequency = epoch_frequency


class SeleKTProcessor:
    """
    SeleKT algorithm processor for LoRA parameters
    
    Features:
    - DDP compatible
    - Memory efficient
    - Specialized for LoRA parameter sparsification
    """
    
    def __init__(self, config: SeleKTConfig):
        self.config = config
        self.step_count = 0
        
    @torch.no_grad()
    def apply_selekt(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Apply SeleKT algorithm to LoRA parameters
        
        Args:
            model: Current model with LoRA parameters
            
        Returns:
            Model with SeleKT applied to LoRA parameters
        """
        if not self.config.enabled:
            return model
            
        logger.info_rank0(f"Applying SeleKT with α={self.config.alpha}")
        
        # Get current rank for DDP coordination
        rank = 0
        if dist.is_initialized():
            rank = dist.get_rank()
        
        # Find all LoRA parameters
        lora_params = []
        for name, param in model.named_parameters():
            if param.requires_grad and self._is_lora_parameter(name):
                lora_params.append((name, param))
        
        logger.info_rank0(f"🎯 Found {len(lora_params)} LoRA parameters to process")
        
        if len(lora_params) == 0:
            logger.warning_rank0("⚠️ No LoRA parameters found, skipping SeleKT application")
            return model
        
        # Process LoRA parameters
        processed_params = 0
        for name, param in tqdm(lora_params, desc="SeleKT-LoRA", disable=(rank != 0)):
            try:
                # Apply selection mask to LoRA parameter
                # LoRA parameters represent learned deltas, so we can directly sparsify them
                mask = self._create_selection_mask(param.data, self.config.alpha)
                param.data = param.data * mask
                
                processed_params += 1
                
            except Exception as e:
                logger.warning_rank0(f"⚠️ Failed to process {name}: {e}")
                continue
        
        logger.info_rank0(f"✅ SeleKT processed {processed_params}/{len(lora_params)} LoRA parameters")
        
        # Memory cleanup
        if self.config.memory_efficient:
            torch.cuda.empty_cache()
            gc.collect()
        
        if dist.is_initialized():
            dist.barrier()
            
        logger.info_rank0("✅ SeleKT application completed")
        return model
    
    def _is_lora_parameter(self, param_name: str) -> bool:
        """Check if parameter is a LoRA parameter"""
        lora_indicators = [
            'lora_a', 'lora_b', 'lora_A', 'lora_B',
            'adapter', 'adapters',
            'peft'  # PEFT library parameters
        ]
        
        param_name_lower = param_name.lower()
        return any(indicator in param_name_lower for indicator in lora_indicators)
    
    def _create_selection_mask(self, tensor: torch.Tensor, alpha: float) -> torch.Tensor:
        """Create selection mask for top-k parameters"""
        if alpha >= 1.0:
            return torch.ones_like(tensor)
        if alpha <= 0.0:
            return torch.zeros_like(tensor)
            
        # Calculate number of parameters to keep
        total_elements = tensor.numel()
        k = max(1, int(alpha * total_elements))
        
        # Create mask for top-k absolute values
        mask = torch.zeros_like(tensor)
        _, indices = torch.topk(tensor.abs().view(-1), k)
        mask.view(-1)[indices] = 1
        
        return mask


class SeleKTCallback:
    """
    SeleKT callback for integration with training loops
    Compatible with both HuggingFace Trainer and custom training
    """
    
    def __init__(self, config: SeleKTConfig):
        self.config = config
        self.processor = SeleKTProcessor(config)
        self.save_count = 0
        self.last_applied_step = 0
        self.last_applied_save = 0
        self.epoch_count = 0
        self.last_applied_epoch = 0
    
    def on_save(self, model: torch.nn.Module, save_path: str = None, current_step: int = None) -> torch.nn.Module:
        """Apply SeleKT when model is saved"""
        if not self.config.apply_on_save:
            return model
            
        self.save_count += 1
        
        # Check whether to apply based on save frequency
        should_apply_by_save = self.save_count % self.config.apply_frequency == 0
        
        # Check whether to apply based on step frequency
        should_apply_by_step = False
        if current_step is not None and self.config.step_frequency is not None:
            steps_since_last = current_step - self.last_applied_step
            should_apply_by_step = steps_since_last >= self.config.step_frequency
        
        # Check whether to apply based on max interval
        should_apply_by_max_interval = False
        if current_step is not None and self.config.max_interval_steps is not None:
            steps_since_last = current_step - self.last_applied_step
            should_apply_by_max_interval = steps_since_last >= self.config.max_interval_steps
        
        # Decide whether to apply SeleKT
        should_apply = should_apply_by_save or should_apply_by_step or should_apply_by_max_interval
        
        if not should_apply:
            return model
            
        # Record trigger reasons
        reasons = []
        if should_apply_by_save:
            reasons.append(f"save_frequency({self.config.apply_frequency})")
        if should_apply_by_step:
            reasons.append(f"step_frequency({self.config.step_frequency})")
        if should_apply_by_max_interval:
            reasons.append(f"max_interval({self.config.max_interval_steps})")
        
        logger.info_rank0(f"Applying SeleKT at save #{self.save_count} (step {current_step})")
        logger.info_rank0(f"Trigger reasons: {', '.join(reasons)}")
        
        # Apply SeleKT to LoRA parameters
        selekt_model = self.processor.apply_selekt(model)
        
        # Update application record
        if current_step is not None:
            self.last_applied_step = current_step
        self.last_applied_save = self.save_count
        
        # Save SeleKT checkpoint if requested
        if self.config.save_selekt_checkpoints and save_path:
            selekt_path = f"{save_path}-selekt"
            self._save_selekt_checkpoint(selekt_model, selekt_path)
            
        return selekt_model
    
    def on_step(self, model: torch.nn.Module, current_step: int) -> torch.nn.Module:
        """Apply SeleKT based on step frequency (independent of saves)"""
        if not self.config.enabled or not self.config.step_frequency:
            return model
            
        steps_since_last = current_step - self.last_applied_step
        
        # Check whether to apply based on step frequency
        should_apply_by_step = steps_since_last >= self.config.step_frequency
        
        # Check whether to apply based on max interval
        should_apply_by_max_interval = False
        if self.config.max_interval_steps is not None:
            should_apply_by_max_interval = steps_since_last >= self.config.max_interval_steps
        
        if not (should_apply_by_step or should_apply_by_max_interval):
            return model
            
        # Record trigger reasons
        reasons = []
        if should_apply_by_step:
            reasons.append(f"step_frequency({self.config.step_frequency})")
        if should_apply_by_max_interval:
            reasons.append(f"max_interval({self.config.max_interval_steps})")
        
        logger.info_rank0(f"Applying SeleKT at step {current_step} (independent of save)")
        logger.info_rank0(f"Trigger reasons: {', '.join(reasons)}")
        
        # Apply SeleKT to LoRA parameters
        selekt_model = self.processor.apply_selekt(model)
        
        # Update application record
        self.last_applied_step = current_step
        
        return selekt_model
    def _save_selekt_checkpoint(self, model: torch.nn.Module, path: str):
        """Save SeleKT processed model"""
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            
            # Save model state
            if hasattr(model, 'save_pretrained'):
                model.save_pretrained(path)
            else:
                torch.save(model.state_dict(), f"{path}/pytorch_model.bin")
                
            logger.info_rank0(f"SeleKT checkpoint saved to: {path}")
            
        except Exception as e:
            logger.error_rank0(f"Failed to save SeleKT checkpoint: {e}")

    def on_epoch_end(self, model: torch.nn.Module, current_epoch: int, current_step: int = None) -> torch.nn.Module:
        """Apply SeleKT at the end of epoch"""
        if not self.config.enabled or not self.config.epoch_frequency:
            return model
            
        self.epoch_count = current_epoch
        epochs_since_last = current_epoch - self.last_applied_epoch
        
        # 检查是否应该基于 epoch 频率应用
        should_apply_by_epoch = epochs_since_last >= self.config.epoch_frequency
        
        # 检查是否超过最大间隔（如果设置了步数限制）
        should_apply_by_max_interval = False
        if current_step is not None and self.config.max_interval_steps is not None:
            steps_since_last = current_step - self.last_applied_step
            should_apply_by_max_interval = steps_since_last >= self.config.max_interval_steps
        
        if not (should_apply_by_epoch or should_apply_by_max_interval):
            return model
            
        # 记录触发原因
        reasons = []
        if should_apply_by_epoch:
            reasons.append(f"epoch_frequency({self.config.epoch_frequency})")
        if should_apply_by_max_interval:
            reasons.append(f"max_interval({self.config.max_interval_steps})")
        
        logger.info_rank0(f"Applying SeleKT at epoch {current_epoch} end")
        logger.info_rank0(f"Trigger reasons: {', '.join(reasons)}")
        
        # Apply SeleKT to LoRA parameters
        selekt_model = self.processor.apply_selekt(model)
        
        # 更新应用记录
        self.last_applied_epoch = current_epoch
        if current_step is not None:
            self.last_applied_step = current_step
        
        return selekt_model


def create_selekt_config(
    alpha: float = 0.05,
    step_frequency: Optional[int] = None,
    max_interval_steps: Optional[int] = None,
    epoch_frequency: Optional[int] = None,
    **kwargs
) -> SeleKTConfig:
    """Factory function to create SeleKT configuration"""
    return SeleKTConfig(
        alpha=alpha,
        step_frequency=step_frequency,
        max_interval_steps=max_interval_steps,
        epoch_frequency=epoch_frequency,
        **kwargs
    )


def apply_selekt_to_model(
    model: torch.nn.Module,
    alpha: float = 0.05
) -> torch.nn.Module:
    """
    Standalone function to apply SeleKT to LoRA parameters
    
    Usage:
        model = apply_selekt_to_model(
            model=trained_model,
            alpha=0.05
        )
    """
    config = SeleKTConfig(
        alpha=alpha,
        enabled=True
    )
    
    processor = SeleKTProcessor(config)
    return processor.apply_selekt(model)