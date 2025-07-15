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
    ):
        self.alpha = alpha
        self.enabled = enabled
        self.save_selekt_checkpoints = save_selekt_checkpoints
        self.apply_on_save = apply_on_save
        self.apply_frequency = apply_frequency
        self.memory_efficient = memory_efficient


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
    
    def on_save(self, model: torch.nn.Module, save_path: str = None) -> torch.nn.Module:
        """Apply SeleKT when model is saved"""
        if not self.config.apply_on_save:
            return model
            
        self.save_count += 1
        
        # Check if should apply based on frequency
        if self.save_count % self.config.apply_frequency != 0:
            return model
            
        logger.info_rank0(f"Applying SeleKT at save #{self.save_count}")
        
        # Apply SeleKT to LoRA parameters
        selekt_model = self.processor.apply_selekt(model)
        
        # Save SeleKT checkpoint if requested
        if self.config.save_selekt_checkpoints and save_path:
            selekt_path = f"{save_path}-selekt"
            self._save_selekt_checkpoint(selekt_model, selekt_path)
            
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


def create_selekt_config(
    alpha: float = 0.05,
    **kwargs
) -> SeleKTConfig:
    """Factory function to create SeleKT configuration"""
    return SeleKTConfig(
        alpha=alpha,
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