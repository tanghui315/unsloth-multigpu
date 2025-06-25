"""
Multi-GPU Manager - System Core Component
Responsible for model replica management, gradient aggregation, and parameter synchronization across GPUs
"""

import copy
import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class MultiGPUManager:
    """
    Multi-GPU Manager - System Core Component
    
    Responsible for:
    1. Creating and managing model replicas on multiple GPUs
    2. Managing optimizers for each GPU
    3. Gradient aggregation and distribution
    4. Model parameter synchronization
    5. Performance monitoring
    """
    
    def __init__(self, base_model: nn.Module, num_gpus: int, optimizer_config: Dict[str, Any]):
        """
        Initialize Multi-GPU Manager
        
        Args:
            base_model: Base model (usually on GPU 0)
            num_gpus: Number of GPUs to use
            optimizer_config: Optimizer configuration {'class': optimizer_class, 'kwargs': {...}}
        """
        self.base_model = base_model
        self.num_gpus = num_gpus
        self.devices = [f"cuda:{i}" for i in range(num_gpus)]
        self.master_gpu = 0  # Main GPU, used for parameter synchronization
        
        # Store models and optimizers on each GPU
        self.device_models: Dict[int, nn.Module] = {}
        self.device_optimizers: Dict[int, torch.optim.Optimizer] = {}
        
        # Optimizer configuration
        self.optimizer_class = optimizer_config['class']
        self.optimizer_kwargs = optimizer_config['kwargs']
        
        # Performance statistics
        self.step_count = 0
        self.total_sync_time = 0.0
        self.total_aggregation_time = 0.0
        
        # Status flag
        self._setup_complete = False
        
        logger.info(f"🔧 Initializing MultiGPUManager: {num_gpus} GPUs, Main GPU: {self.master_gpu}")
    
    def setup_device_models(self):
        """
        Create model replicas and optimizers on each GPU
        This is a heavy operation, should be executed only once
        """
        if self._setup_complete:
            logger.warning("Device models already set up, skipping repeated setup")
            return
            
        logger.info(f"Starting to set up model replicas on {self.num_gpus} GPUs...")
        
        start_time = time.time()
        
        # ---------- Plan A: Move the original model back to CPU first to avoid two copies of weights on GPU-0 ----------
        import torch  # Local import to avoid circular dependency
        self.base_model.to("cpu")
        # Immediately release the original main GPU cache to avoid reserved memory still being occupied after to(cpu)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for gpu_id in range(self.num_gpus):
            device = f"cuda:{gpu_id}"
            
            with torch.cuda.device(gpu_id):
                logger.info(f"📦 Creating model replica on GPU {gpu_id}...")
                
                # Deep copy model to specified GPU
                if gpu_id == self.master_gpu:
                    # The main GPU uses the original model (only move once, non_blocking for further acceleration)
                    model_copy = self.base_model.to(device, non_blocking=True)
                else:
                    # Other GPUs create deep copy
                    model_copy = copy.deepcopy(self.base_model).to(device)
                
                self.device_models[gpu_id] = model_copy
                
                # Create optimizer for each model replica
                optimizer = self.optimizer_class(
                    model_copy.parameters(), 
                    **self.optimizer_kwargs
                )
                self.device_optimizers[gpu_id] = optimizer
                
                # Check GPU memory usage
                allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3  # GB
                cached = torch.cuda.memory_reserved(gpu_id) / 1024**3  # GB
                logger.info(f"   GPU {gpu_id} Memory Usage: {allocated:.2f}GB allocated, {cached:.2f}GB cached")
        
        # ---------- Plan A: After the loop, release base_model on CPU to reduce extra usage ----------
        del self.base_model
        torch.cuda.empty_cache()

        setup_time = time.time() - start_time
        self._setup_complete = True
        
        logger.info(f"✅ Model replicas set up, time taken: {setup_time:.2f} seconds")
        
        # Validate setup
        self._validate_setup()
    
    def _validate_setup(self):
        """Validate the correctness of model setup"""
        logger.info("🔍 Validating model setup...")
        
        master_model = self.device_models[self.master_gpu]
        master_param_count = sum(p.numel() for p in master_model.parameters())
        
        for gpu_id in range(self.num_gpus):
            if gpu_id == self.master_gpu:
                continue
                
            model = self.device_models[gpu_id]
            param_count = sum(p.numel() for p in model.parameters())
            
            if param_count != master_param_count:
                raise RuntimeError(f"GPU {gpu_id} Model parameter count mismatch: {param_count} vs {master_param_count}")
                
        logger.info(f"✅ Model validation passed, each model parameter count: {master_param_count:,}")
    
    @contextmanager
    def performance_timer(self, operation_name: str):
        """Performance timing context manager"""
        # Check if CUDA is available
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            start_time = time.time()
            
            yield
            
            end_event.record()
            torch.cuda.synchronize()
            
            gpu_time = start_event.elapsed_time(end_event)  # ms
            wall_time = (time.time() - start_time) * 1000  # ms
            
            logger.debug(f"⏱️ {operation_name}: GPU time {gpu_time:.2f}ms, Total time {wall_time:.2f}ms")
        else:
            # CPU environment, use wall time only
            start_time = time.time()
            
            yield
            
            wall_time = (time.time() - start_time) * 1000  # ms
            logger.debug(f"⏱️ {operation_name}: Total time {wall_time:.2f}ms (CPU mode)")
    
    def aggregate_gradients(self, gradients_per_gpu: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Efficient gradient aggregation algorithm
        
        Args:
            gradients_per_gpu: List of gradient dictionaries per GPU
            
        Returns:
            Aggregated gradient dictionary
        """
        if not gradients_per_gpu:
            return {}
            
        with self.performance_timer("Gradient Aggregation"):
            aggregated = {}
            
            # Use gradients from the first GPU as a baseline
            master_gradients = gradients_per_gpu[0]
            
            for param_name, grad in master_gradients.items():
                if grad is None:
                    continue
                    
                # Sum gradients from all GPUs
                total_grad = grad.clone()
                
                for gpu_id in range(1, len(gradients_per_gpu)):
                    if param_name in gradients_per_gpu[gpu_id]:
                        other_grad = gradients_per_gpu[gpu_id][param_name]
                        if other_grad is not None:
                            # Transfer gradients to main GPU for aggregation
                            total_grad += other_grad.to(grad.device, non_blocking=True)
                
                # Average gradients
                aggregated[param_name] = total_grad / len(gradients_per_gpu)
        
        self.total_aggregation_time += time.time()
        return aggregated
    
    def apply_aggregated_gradients(self, aggregated_gradients: Dict[str, torch.Tensor]):
        """
        Apply aggregated gradients to all model replicas on all GPUs
        
        Args:
            aggregated_gradients: Aggregated gradient dictionary
        """
        with self.performance_timer("Gradient Application"):
            for gpu_id in range(self.num_gpus):
                model = self.device_models[gpu_id]
                optimizer = self.device_optimizers[gpu_id]
                
                with torch.cuda.device(gpu_id):
                    # Set aggregated gradients to model parameters
                    for name, param in model.named_parameters():
                        if name in aggregated_gradients and param.requires_grad:
                            # Transfer gradients to corresponding device
                            param.grad = aggregated_gradients[name].to(param.device, non_blocking=True)
                    
                    # Execute optimizer step
                    optimizer.step()
                    optimizer.zero_grad()
        
        self.step_count += 1
    
    def sync_model_parameters(self, force_sync: bool = False):
        """
        Synchronize model parameters across all GPUs (from main GPU to other GPUs)
        
        Args:
            force_sync: Whether to force synchronization, default False (will be decided based on step frequency)
        """
        if self.num_gpus <= 1:
            return
            
        # Default sync every 10 steps to reduce communication overhead
        if not force_sync and self.step_count % 10 != 0:
            return
            
        with self.performance_timer("Parameter Synchronization"):
            master_model = self.device_models[self.master_gpu]
            
            for gpu_id in range(self.num_gpus):
                if gpu_id == self.master_gpu:
                    continue
                    
                target_model = self.device_models[gpu_id]
                
                with torch.cuda.device(gpu_id):
                    for master_param, target_param in zip(
                        master_model.parameters(), 
                        target_model.parameters()
                    ):
                        target_param.data.copy_(
                            master_param.data.to(target_param.device, non_blocking=True)
                        )
        
        self.total_sync_time += time.time()
    
    def get_model(self, gpu_id: int) -> nn.Module:
        """Get model on specified GPU"""
        if gpu_id not in self.device_models:
            raise ValueError(f"No model on GPU {gpu_id}")
        return self.device_models[gpu_id]
    
    def get_optimizer(self, gpu_id: int) -> torch.optim.Optimizer:
        """Get optimizer on specified GPU"""
        if gpu_id not in self.device_optimizers:
            raise ValueError(f"No optimizer on GPU {gpu_id}")
        return self.device_optimizers[gpu_id]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        avg_sync_time = self.total_sync_time / max(self.step_count, 1)
        avg_aggregation_time = self.total_aggregation_time / max(self.step_count, 1)
        
        return {
            'total_steps': self.step_count,
            'avg_sync_time_ms': avg_sync_time * 1000,
            'avg_aggregation_time_ms': avg_aggregation_time * 1000,
            'num_gpus': self.num_gpus,
            'setup_complete': self._setup_complete
        }
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("🧹 Cleaning up MultiGPUManager resources...")
        
        for gpu_id in range(self.num_gpus):
            with torch.cuda.device(gpu_id):
                if gpu_id in self.device_models:
                    del self.device_models[gpu_id]
                if gpu_id in self.device_optimizers:
                    del self.device_optimizers[gpu_id]
                torch.cuda.empty_cache()
        
        self.device_models.clear()
        self.device_optimizers.clear()
        self._setup_complete = False
        
        logger.info("✅ Resources cleaned up")
    
    def __del__(self):
        """Destructor, ensure resources cleaned up"""
        if hasattr(self, '_setup_complete') and self._setup_complete:
            self.cleanup() 