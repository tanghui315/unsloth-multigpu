"""
Gradient Aggregator - Ensures Training Correctness
Responsible for efficient gradient aggregation, consistency validation, and communication optimization
"""

import logging
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

class AggregationMethod(Enum):
    """梯度聚合方法枚举"""
    MEAN = "mean"           # 平均值（标准数据并行）
    SUM = "sum"             # 求和
    WEIGHTED_MEAN = "weighted_mean"  # 加权平均
    MEDIAN = "median"       # 中位数（对异常值鲁棒）

class GradientAggregator:
    """
    Gradient Aggregator - Ensures Training Correctness
    
    Responsible for:
    1. Efficient gradient aggregation algorithm
    2. Gradient consistency validation
    3. Communication overhead optimization
    4. Numerical stability guarantee
    5. Exception gradient detection
    """
    
    def __init__(self, 
                 aggregation_method: AggregationMethod = AggregationMethod.MEAN,
                 enable_consistency_check: bool = True,
                 consistency_tolerance: float = 1e-3,
                 enable_gradient_clipping: bool = True,
                 max_grad_norm: float = 1.0):
        """
        Initialize Gradient Aggregator
        
        Args:
            aggregation_method: Aggregation method
            enable_consistency_check: Whether to enable consistency check
            consistency_tolerance: Consistency tolerance
            enable_gradient_clipping: Whether to enable gradient clipping
            max_grad_norm: Maximum gradient norm
        """
        self.aggregation_method = aggregation_method
        self.enable_consistency_check = enable_consistency_check
        self.consistency_tolerance = consistency_tolerance
        self.enable_gradient_clipping = enable_gradient_clipping
        self.max_grad_norm = max_grad_norm
        
        # 统计信息
        self.total_aggregations = 0
        self.total_aggregation_time = 0.0
        self.consistency_failures = 0
        self.gradient_clips = 0
        
        # 梯度统计
        self.gradient_norms_history = []
        self.max_gradient_norms = {}
        
        logger.info(f"🔧 Initialize GradientAggregator: method={aggregation_method.value}, "
                   f"consistency check={'enabled' if enable_consistency_check else 'disabled'}")
    
    def aggregate(self, 
                  gradients_per_gpu: List[Dict[str, torch.Tensor]], 
                  gpu_weights: Optional[List[float]] = None) -> Dict[str, torch.Tensor]:
        """
        Execute gradient aggregation
        
        Args:
            gradients_per_gpu: List of gradient dictionaries per GPU
            gpu_weights: GPU weights (for weighted aggregation)
            
        Returns:
            Aggregated gradient dictionary
        """
        if not gradients_per_gpu:
            logger.warning("Received empty gradient list")
            return {}
        
        start_time = time.time()
        
        # Filter out empty gradients
        valid_gradients = [grad_dict for grad_dict in gradients_per_gpu if grad_dict]
        
        if not valid_gradients:
            logger.warning("All gradient dictionaries are empty")
            return {}
        
        logger.debug(f"📊 Aggregating gradients from {len(valid_gradients)} GPUs")
        
        # Consistency check (before aggregation)
        if self.enable_consistency_check:
            self._validate_gradient_structure(valid_gradients)
        
        # Execute aggregation
        aggregated_gradients = self._perform_aggregation(valid_gradients, gpu_weights)
        
        # Gradient clipping
        if self.enable_gradient_clipping:
            aggregated_gradients = self._clip_gradients(aggregated_gradients)
        
        # Record statistics
        aggregation_time = time.time() - start_time
        self.total_aggregation_time += aggregation_time
        self.total_aggregations += 1
        
        # Record gradient norms
        self._record_gradient_norms(aggregated_gradients)
        
        logger.debug(f"✅ Gradient aggregation completed, time: {aggregation_time*1000:.2f}ms")
        
        return aggregated_gradients
    
    def _validate_gradient_structure(self, gradients_per_gpu: List[Dict[str, torch.Tensor]]):
        """Validate gradient structure consistency"""
        if len(gradients_per_gpu) < 2:
            return
        
        reference_grad = gradients_per_gpu[0]
        reference_keys = set(reference_grad.keys())
        
        for gpu_id, grad_dict in enumerate(gradients_per_gpu[1:], 1):
            current_keys = set(grad_dict.keys())
            
            # Check key consistency
            if current_keys != reference_keys:
                missing_keys = reference_keys - current_keys
                extra_keys = current_keys - reference_keys
                
                error_msg = f"GPU {gpu_id} gradient keys are inconsistent"
                if missing_keys:
                    error_msg += f", missing: {missing_keys}"
                if extra_keys:
                    error_msg += f", extra: {extra_keys}"
                
                logger.error(error_msg)
                self.consistency_failures += 1
                
                # Raise exception in strict mode
                if self.consistency_tolerance == 0:
                    raise ValueError(error_msg)
            
            # Check tensor shape consistency
            for key in reference_keys.intersection(current_keys):
                ref_tensor = reference_grad[key]
                cur_tensor = grad_dict[key]
                
                if ref_tensor is None or cur_tensor is None:
                    continue
                
                if ref_tensor.shape != cur_tensor.shape:
                    error_msg = f"GPU {gpu_id} parameter {key} shape mismatch: {ref_tensor.shape} vs {cur_tensor.shape}"
                    logger.error(error_msg)
                    self.consistency_failures += 1
                    
                    if self.consistency_tolerance == 0:
                        raise ValueError(error_msg)
    
    def _perform_aggregation(self, 
                           gradients_per_gpu: List[Dict[str, torch.Tensor]], 
                           gpu_weights: Optional[List[float]] = None) -> Dict[str, torch.Tensor]:
        """Execute specific aggregation algorithm"""
        
        if self.aggregation_method == AggregationMethod.MEAN:
            return self._aggregate_mean(gradients_per_gpu)
        
        elif self.aggregation_method == AggregationMethod.SUM:
            return self._aggregate_sum(gradients_per_gpu)
        
        elif self.aggregation_method == AggregationMethod.WEIGHTED_MEAN:
            return self._aggregate_weighted_mean(gradients_per_gpu, gpu_weights)
        
        elif self.aggregation_method == AggregationMethod.MEDIAN:
            return self._aggregate_median(gradients_per_gpu)
        
        else:
            raise ValueError(f"Unsupported aggregation method: {self.aggregation_method}")
    
    def _aggregate_mean(self, gradients_per_gpu: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Average aggregation (standard data parallel)"""
        aggregated = {}
        num_gpus = len(gradients_per_gpu)
        
        # Use the gradient from the first GPU as a reference
        reference_grad = gradients_per_gpu[0]
        
        for param_name, grad in reference_grad.items():
            if grad is None:
                aggregated[param_name] = None
                continue
            
            # Sum all gradients from all GPUs
            total_grad = grad.clone()
            
            for gpu_id in range(1, num_gpus):
                if param_name in gradients_per_gpu[gpu_id]:
                    other_grad = gradients_per_gpu[gpu_id][param_name]
                    if other_grad is not None:
                        # Ensure calculation on the same device
                        total_grad += other_grad.to(grad.device, non_blocking=True)
            
            # Average
            aggregated[param_name] = total_grad / num_gpus
        
        return aggregated
    
    def _aggregate_sum(self, gradients_per_gpu: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Sum aggregation"""
        aggregated = {}
        reference_grad = gradients_per_gpu[0]
        
        for param_name, grad in reference_grad.items():
            if grad is None:
                aggregated[param_name] = None
                continue
            
            total_grad = grad.clone()
            
            for gpu_id in range(1, len(gradients_per_gpu)):
                if param_name in gradients_per_gpu[gpu_id]:
                    other_grad = gradients_per_gpu[gpu_id][param_name]
                    if other_grad is not None:
                        total_grad += other_grad.to(grad.device, non_blocking=True)
            
            aggregated[param_name] = total_grad
        
        return aggregated
    
    def _aggregate_weighted_mean(self, 
                               gradients_per_gpu: List[Dict[str, torch.Tensor]], 
                               weights: Optional[List[float]] = None) -> Dict[str, torch.Tensor]:
        """Weighted average aggregation"""
        if weights is None:
            # Fallback to regular average if no weights are provided
            return self._aggregate_mean(gradients_per_gpu)
        
        if len(weights) != len(gradients_per_gpu):
            raise ValueError(f"Weight count {len(weights)} does not match GPU count {len(gradients_per_gpu)}")
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        aggregated = {}
        reference_grad = gradients_per_gpu[0]
        
        for param_name, grad in reference_grad.items():
            if grad is None:
                aggregated[param_name] = None
                continue
            
            # Weighted sum
            weighted_grad = grad.clone() * normalized_weights[0]
            
            for gpu_id in range(1, len(gradients_per_gpu)):
                if param_name in gradients_per_gpu[gpu_id]:
                    other_grad = gradients_per_gpu[gpu_id][param_name]
                    if other_grad is not None:
                        weighted_grad += (other_grad.to(grad.device, non_blocking=True) 
                                        * normalized_weights[gpu_id])
            
            aggregated[param_name] = weighted_grad
        
        return aggregated
    
    def _aggregate_median(self, gradients_per_gpu: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Median aggregation (robust to outliers)"""
        aggregated = {}
        reference_grad = gradients_per_gpu[0]
        
        for param_name, grad in reference_grad.items():
            if grad is None:
                aggregated[param_name] = None
                continue
            
            # Collect all gradients for this parameter from all GPUs
            grad_tensors = []
            
            for gpu_id in range(len(gradients_per_gpu)):
                if param_name in gradients_per_gpu[gpu_id]:
                    other_grad = gradients_per_gpu[gpu_id][param_name]
                    if other_grad is not None:
                        grad_tensors.append(other_grad.to(grad.device, non_blocking=True))
            
            if grad_tensors:
                # Stack along new dimension, then calculate median
                stacked = torch.stack(grad_tensors, dim=0)
                median_grad = torch.median(stacked, dim=0)[0]
                aggregated[param_name] = median_grad
            else:
                aggregated[param_name] = None
        
        return aggregated
    
    def _clip_gradients(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Gradient clipping"""
        if not gradients:
            return gradients
        
        # Calculate total gradient norm
        total_norm = 0.0
        param_norms = []
        
        for param_name, grad in gradients.items():
            if grad is not None:
                param_norm = grad.norm().item()
                param_norms.append(param_norm)
                total_norm += param_norm ** 2
        
        total_norm = total_norm ** 0.5
        
        # Check if clipping is needed
        if total_norm > self.max_grad_norm:
            clip_ratio = self.max_grad_norm / total_norm
            
            # Scale all gradients proportionally
            for param_name, grad in gradients.items():
                if grad is not None:
                    gradients[param_name] = grad * clip_ratio
            
            self.gradient_clips += 1
            logger.debug(f"🔧 Gradient clipping: original norm={total_norm:.4f}, clipping ratio={clip_ratio:.4f}")
        
        return gradients
    
    def _record_gradient_norms(self, gradients: Dict[str, torch.Tensor]):
        """Record gradient norm statistics"""
        total_norm = 0.0
        
        for param_name, grad in gradients.items():
            if grad is not None:
                param_norm = grad.norm().item()
                total_norm += param_norm ** 2
                
                # Record single parameter's maximum norm
                if param_name not in self.max_gradient_norms:
                    self.max_gradient_norms[param_name] = param_norm
                else:
                    self.max_gradient_norms[param_name] = max(
                        self.max_gradient_norms[param_name], param_norm
                    )
        
        total_norm = total_norm ** 0.5
        self.gradient_norms_history.append(total_norm)
        
        # Maintain history record size
        if len(self.gradient_norms_history) > 1000:
            self.gradient_norms_history = self.gradient_norms_history[-1000:]
    
    def validate_consistency(self, 
                           gradients_before: Dict[str, torch.Tensor], 
                           gradients_after: Dict[str, torch.Tensor]) -> bool:
        """Validate gradient consistency (before and after aggregation)"""
        if not self.enable_consistency_check:
            return True
        
        if not gradients_before or not gradients_after:
            logger.warning("Gradient is empty, skipping consistency check")
            return True
        
        inconsistencies = []
        
        for param_name in gradients_before:
            if param_name not in gradients_after:
                inconsistencies.append(f"Parameter {param_name} disappeared after aggregation")
                continue
            
            grad_before = gradients_before[param_name]
            grad_after = gradients_after[param_name]
            
            if grad_before is None and grad_after is None:
                continue
            
            if grad_before is None or grad_after is None:
                inconsistencies.append(f"Parameter {param_name} gradient None state inconsistency")
                continue
            
            # Check shape
            if grad_before.shape != grad_after.shape:
                inconsistencies.append(f"Parameter {param_name} shape mismatch: {grad_before.shape} vs {grad_after.shape}")
                continue
            
            # Check numerical difference
            relative_error = torch.abs(grad_after - grad_before) / (torch.abs(grad_before) + 1e-8)
            max_error = torch.max(relative_error).item()
            
            if max_error > self.consistency_tolerance:
                inconsistencies.append(f"Parameter {param_name} relative error too large: {max_error:.6f}")
        
        if inconsistencies:
            logger.warning(f"Found {len(inconsistencies)} consistency issues:")
            for issue in inconsistencies[:5]:  # Only display first 5 issues
                logger.warning(f"  - {issue}")
            if len(inconsistencies) > 5:
                logger.warning(f"  - ... and {len(inconsistencies) - 5} more issues")
            
            self.consistency_failures += 1
            return False
        
        return True
    
    def optimize_communication(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Optimize cross-GPU communication (e.g., compression, quantization, etc.)"""
        # Here you can implement gradient compression, quantization, etc. optimization techniques
        # For now, simply return the original gradients
        return gradients
    
    def get_aggregation_stats(self) -> Dict[str, Any]:
        """Get aggregation statistics"""
        avg_aggregation_time = (self.total_aggregation_time / 
                               max(self.total_aggregations, 1))
        
        gradient_norm_stats = {}
        if self.gradient_norms_history:
            gradient_norm_stats = {
                'mean_norm': np.mean(self.gradient_norms_history),
                'max_norm': np.max(self.gradient_norms_history),
                'min_norm': np.min(self.gradient_norms_history),
                'std_norm': np.std(self.gradient_norms_history)
            }
        
        return {
            'total_aggregations': self.total_aggregations,
            'avg_aggregation_time_ms': avg_aggregation_time * 1000,
            'consistency_failures': self.consistency_failures,
            'gradient_clips': self.gradient_clips,
            'aggregation_method': self.aggregation_method.value,
            'gradient_norm_stats': gradient_norm_stats,
            'max_param_norms': dict(list(self.max_gradient_norms.items())[:10])  # Only display first 10
        }
    
    def reset_stats(self):
        """Reset statistics"""
        self.total_aggregations = 0
        self.total_aggregation_time = 0.0
        self.consistency_failures = 0
        self.gradient_clips = 0
        self.gradient_norms_history.clear()
        self.max_gradient_norms.clear()
        
        logger.info("📊 Gradient aggregation statistics have been reset") 