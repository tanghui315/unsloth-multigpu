"""
Multi-GPU Trainer - Full Training Loop Implementation
Responsible for coordinating the complete process of multi-GPU parallel training
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler
from tqdm.auto import tqdm

from ..utils import MultiGPUConfig
from .batch_sharding import BatchSharding
from .gradient_aggregator import AggregationMethod, GradientAggregator
from .multi_gpu_manager import MultiGPUManager

logger = logging.getLogger(__name__)


class MultiGPUTrainer:
    """
    Multi-GPU Trainer - Full Training Loop Implementation
    
    By wrapping the original HuggingFace Trainer, it implements data parallel training.
    """
    
    def __init__(self, 
                 original_trainer: Any,
                 config: MultiGPUConfig):
        """
        Initialize Multi-GPU Trainer
        
        Args:
            original_trainer: Original HuggingFace Trainer object
            config: Multi-GPU training configuration object
        """
        if not hasattr(original_trainer, 'model'):
            raise ValueError("Original Trainer must contain a 'model' attribute")
        
        self.original_trainer = original_trainer
        self.config = config
        self.model = original_trainer.model
        
        # Extract optimizer configuration from original trainer
        optimizer_config = self._extract_optimizer_config(original_trainer)
        
        # Initialize core components
        self.multi_gpu_manager = MultiGPUManager(
            self.model, 
            self.config.num_gpus, 
            optimizer_config
        )
        self.batch_sharder = BatchSharding(self.config.num_gpus)
        self.gradient_aggregator = GradientAggregator(
            aggregation_method=self._get_aggregation_method(self.config.gradient_aggregation),
            enable_consistency_check=True,  # Can be configured in config
            enable_gradient_clipping=True,  # Can be configured in config
            max_grad_norm=1.0               # Can be configured in config
        )
        
        # Mixed precision support
        self.autocast_dtype = next(self.model.parameters()).dtype if torch.cuda.is_available() else None
        if self.config.enable_mixed_precision and torch.cuda.is_available():
            # Only needed if dtype is float16; bfloat16 does not need GradScaler.
            if self.autocast_dtype == torch.float16:
                self.scaler = torch.cuda.amp.GradScaler()
            else:
                self.scaler = None
            logger.info(f"✅ Enabled mixed precision training (AMP), dtype={self.autocast_dtype}")
        else:
            self.scaler = None
            
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.training_stats = {
            'total_forward_time': 0.0,
            'total_backward_time': 0.0,
            'total_aggregation_time': 0.0,
            'total_sync_time': 0.0,
            'num_batches_processed': 0,
            'num_samples_processed': 0
        }
        
        # Device model (delayed initialization)
        self._setup_complete = False

        # Initialize gradient accumulation
        self.gradient_accumulation_steps = original_trainer.args.gradient_accumulation_steps
        self.accumulation_step = 0
        self.accumulated_gradients = {}

        logger.info(f"🔧 Initializing MultiGPUTrainer: {self.config.num_gpus} GPUs, aggregation method: {self.config.gradient_aggregation}")
    
    def setup(self):
        """Set up trainer components"""
        if self._setup_complete:
            return
            
        logger.info("🚀 Setting up Multi-GPU Trainer components...")
        
        # Set up GPU model replicas
        self.multi_gpu_manager.setup_device_models()
        
        # Validate setup
        self._validate_setup()
        
        self._setup_complete = True
        logger.info("✅ Multi-GPU Trainer setup completed")
    
    def _validate_setup(self):
        """Validate trainer setup"""
        # Check GPU availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available, cannot perform multi-GPU training")
            
        if torch.cuda.device_count() < self.config.num_gpus:
            raise RuntimeError(f"Available GPU count {torch.cuda.device_count()} < required count {self.config.num_gpus}")
        
        # Check model parameter consistency
        master_model = self.multi_gpu_manager.get_model(0)
        total_params = sum(p.numel() for p in master_model.parameters())
        logger.info(f"✅ Model validation passed: {total_params:,} parameters")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Execute a single training step
        
        Args:
            batch: Training batch data
            
        Returns:
            Dict containing loss and statistics
        """
        if not self._setup_complete:
            self.setup()
            
        step_start_time = time.time()
        
        # 1. Shard batch
        with self._timer("batch_sharding"):
            shards = self.batch_sharder.split_batch(batch)
            
        # 2. Parallel forward and backward propagation
        losses, gradients = self._parallel_forward_backward(shards)
        
        # 3. Gradient aggregation
        with self._timer("gradient_aggregation"):
            aggregated_gradients = self.gradient_aggregator.aggregate(gradients)
            
        # 4. Accumulate gradients
        self._accumulate_gradients(aggregated_gradients)
        self.accumulation_step += 1

        # 5. Apply gradients after reaching gradient accumulation steps
        if self.accumulation_step % self.gradient_accumulation_steps == 0:
            with self._timer("gradient_apply"):
                self._apply_accumulated_gradients()
            # Reset accumulation
            self.accumulation_step = 0
            self.accumulated_gradients = {}
            self.global_step += 1

        # 6. Sync model parameters (if optimization steps occur)
        if self.global_step and self.global_step % self.config.cleanup_interval == 0:
            with self._timer("parameter_sync"):
                self.multi_gpu_manager.sync_model_parameters()

        # 7. Collect output results
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        self.training_stats['num_batches_processed'] += 1
        self.training_stats['num_samples_processed'] += len(batch.get('input_ids', []))

        # Calculate grad_norm (using aggregated gradients, copied to self.accumulated_gradients before optimizer application)
        try:
            grad_norm = torch.sqrt(
                sum(g.float().pow(2).sum() for g in aggregated_gradients.values())
            ).item()
        except Exception:
            grad_norm = 0.0

        # Get current learning rate (main GPU optimizer)
        try:
            lr = self.multi_gpu_manager.get_optimizer(0).param_groups[0].get('lr', 0.0)
        except Exception:
            lr = 0.0

        step_time = time.time() - step_start_time
        return {
            'loss': avg_loss,
            'grad_norm': grad_norm,
            'learning_rate': lr,
            'step_time': step_time,
            'throughput': len(batch.get('input_ids', [])) / step_time if step_time > 0 else 0,
            'global_step': self.global_step,
        }
    
    def _parallel_forward_backward(self, shards: List[Dict[str, torch.Tensor]]) -> Tuple[List[float], List[Dict[str, torch.Tensor]]]:
        """
        Complete parallel forward and backward propagation in one function
        """
        losses = []
        gradients_per_gpu = []

        for gpu_id, shard in enumerate(shards):
            # Handle empty shard - add placeholder instead of skipping
            if not shard or not shard.get('input_ids', torch.empty(0)).numel():
                losses.append(0.0)  # Empty shard loss is 0
                gradients_per_gpu.append({})  # Empty gradient dictionary
                continue

            model = self.multi_gpu_manager.get_model(gpu_id)
            optimizer = self.multi_gpu_manager.get_optimizer(gpu_id)
            device = f"cuda:{gpu_id}"

            with torch.cuda.device(gpu_id):
                gpu_shard = self._move_to_device(shard, device)

                # Zero gradients (before forward propagation)
                optimizer.zero_grad()

                # Forward propagation
                if self.config.enable_mixed_precision and self.autocast_dtype in (torch.float16, torch.bfloat16):
                    with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                        outputs = self._forward_pass(model, gpu_shard)
                        loss = outputs['loss']
                    # Backward propagation
                    if self.scaler:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                else:
                    outputs = self._forward_pass(model, gpu_shard)
                    loss = outputs['loss']
                    loss.backward()
                
                losses.append(loss.item())
                
                # Collect gradients
                gradients = {}
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        gradients[name] = param.grad.detach().cpu()
                gradients_per_gpu.append(gradients)

        return losses, gradients_per_gpu

    def _parallel_forward(self, shards: List[Dict[str, torch.Tensor]]) -> List[Dict[str, Any]]:
        """
        Parallel forward propagation
        
        Args:
            shards: Batched data split into shards
            
        Returns:
            Forward propagation results for each GPU
        """
        # This method is integrated in _parallel_forward_backward and can be marked as deprecated or internal use
        pass

    def _parallel_backward(self, forward_results: List[Dict[str, Any]]) -> List[Dict[str, torch.Tensor]]:
        """
        Parallel backward propagation
        
        Args:
            forward_results: Forward propagation results
            
        Returns:
            Gradient dictionaries for each GPU
        """
        # This method is integrated in _parallel_forward_backward and can be marked as deprecated or internal use
        pass
    
    def _forward_pass(self, model: nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Execute forward propagation
        
        Args:
            model: Model
            batch: Batch data
            
        Returns:
            Forward propagation output
        """
        # Set to training mode
        model.train()
        
        # Execute forward propagation
        if 'labels' in batch:
            # Supervised learning with labels
            outputs = model(**batch)
            return {
                'loss': outputs.loss,
                'logits': outputs.logits if hasattr(outputs, 'logits') else None
            }
        else:
            # Unlabeled inference
            with torch.no_grad():
                outputs = model(**batch)
                return {
                    'loss': torch.tensor(0.0, device=batch['input_ids'].device),
                    'logits': outputs.logits if hasattr(outputs, 'logits') else outputs
                }
    
    def _move_to_device(self, batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
        """Move batch data to specified device"""
        gpu_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                gpu_batch[key] = value.to(device, non_blocking=True)
            else:
                gpu_batch[key] = value
        return gpu_batch
    
    def _collect_outputs(self, forward_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collect and aggregate forward propagation outputs"""
        # Calculate average loss
        total_loss = sum(r['loss'].item() * r['shard_size'] for r in forward_results if r['shard_size'] > 0)
        total_samples = sum(r['shard_size'] for r in forward_results)
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        
        return {'loss': avg_loss}
    
    def _timer(self, operation: str):
        """Performance timer context manager"""
        class Timer:
            def __init__(self, trainer, op_name):
                self.trainer = trainer
                self.op_name = op_name
                self.start_time = None
                
            def __enter__(self):
                self.start_time = time.time()
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                elapsed = time.time() - self.start_time
                if self.op_name == "forward_pass":
                    self.trainer.training_stats['total_forward_time'] += elapsed
                elif self.op_name == "backward_pass":
                    self.trainer.training_stats['total_backward_time'] += elapsed
                elif self.op_name == "gradient_aggregation":
                    self.trainer.training_stats['total_aggregation_time'] += elapsed
                elif self.op_name == "parameter_sync":
                    self.trainer.training_stats['total_sync_time'] += elapsed
        
        return Timer(self, operation)
    
    def train(self) -> Dict[str, Any]:
        """
        Execute complete training loop
        """
        self.setup()
        
        trainer = self.original_trainer
        args = trainer.args
        
        # Get data loader
        train_dataloader = trainer.get_train_dataloader()
        
        # Calculate total steps
        num_update_steps_per_epoch = len(train_dataloader)
        if args.max_steps > 0:
            max_steps = args.max_steps
            num_train_epochs = args.max_steps // num_update_steps_per_epoch + 1
        else:
            max_steps = num_update_steps_per_epoch * args.num_train_epochs
            num_train_epochs = args.num_train_epochs

        logger.info("***** Multi-GPU Training Begin *****")
        logger.info(f"  Num examples = {len(trainer.train_dataset)}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per GPU = {self.config.batch_size_per_gpu}")
        logger.info(f"  Total train batch size (w. parallel) = {self.config.batch_size_per_gpu * self.config.num_gpus}")
        logger.info(f"  Total optimization steps = {max_steps}")

        start_time = time.time()
        last_step_result = {'loss': 0.0}  # Initialize default value

        # Progress bar
        pbar = tqdm(total=max_steps, desc="Training", dynamic_ncols=True)

        for epoch in range(int(num_train_epochs)):
            self.epoch = epoch
            epoch_iterator = iter(train_dataloader)
            
            for step in range(num_update_steps_per_epoch):
                if self.global_step >= max_steps:
                    break
                
                try:
                    batch = next(epoch_iterator)
                    step_result = self.train_step(batch)
                    last_step_result = step_result  # Save last result
                    
                    if self.global_step % args.logging_steps == 0:
                        epoch_float = epoch + (step / num_update_steps_per_epoch)
                        log_dict = {
                            'loss': step_result['loss'],
                            'grad_norm': step_result['grad_norm'],
                            'learning_rate': step_result['learning_rate'],
                            'epoch': round(epoch_float, 4),
                        }
                        logger.info(log_dict)
                        pbar.set_postfix({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in log_dict.items()})

                    pbar.update(1)

                except StopIteration:
                    break
            
            if self.global_step >= max_steps:
                break

        end_time = time.time()
        
        # Return compatible HuggingFace Trainer statistics
        return {
            "train_runtime": end_time - start_time,
            "train_samples_per_second": self.training_stats['num_samples_processed'] / (end_time - start_time),
            "train_steps_per_second": self.global_step / (end_time - start_time),
            "train_loss": last_step_result['loss'],
            "epoch": self.epoch,
            "global_step": self.global_step,
        }

    def train_epoch(self, dataloader: DataLoader, max_steps: Optional[int] = None) -> Dict[str, Any]:
        """
        Train a single epoch
        [Note] This method will be replaced by new `train` method, temporarily retained for reference.
        
        Args:
            dataloader: Data loader
            max_steps: Maximum training step limit
            
        Returns:
            Epoch training statistics
        """
        epoch_start_time = time.time()
        epoch_losses = []
        epoch_stats = {
            'steps': 0,
            'samples': 0,
            'avg_loss': 0.0,
            'epoch_time': 0.0,
            'avg_step_time': 0.0
        }
        
        logger.info(f"🚀 Starting training Epoch {self.epoch + 1}")
        
        for step, batch in enumerate(dataloader):
            if max_steps and step >= max_steps:
                logger.info(f"⏹️ Reached maximum step limit: {max_steps}")
                break
                
            try:
                # Execute training step
                step_result = self.train_step(batch)
                
                epoch_losses.append(step_result['loss'])
                epoch_stats['steps'] += 1
                epoch_stats['samples'] += step_result.get('num_samples', 0)
                
                # Periodically print progress
                if step % 10 == 0:
                    logger.info(f"  Step {step}: Loss={step_result['loss']:.4f}, "
                              f"Throughput={step_result.get('throughput', 0):.1f} samples/s")
                
            except Exception as e:
                logger.error(f"❌ Training step {step} failed: {e}")
                # Optionally continue or stop training
                raise
        
        # Calculate epoch statistics
        epoch_time = time.time() - epoch_start_time
        epoch_stats.update({
            'avg_loss': sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0,
            'epoch_time': epoch_time,
            'avg_step_time': epoch_time / max(epoch_stats['steps'], 1)
        })
        
        self.epoch += 1
        
        logger.info(f"✅ Epoch {self.epoch} completed: Average loss={epoch_stats['avg_loss']:.4f}, "
                   f"Time={epoch_time:.1f}s")
        
        return epoch_stats
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        stats = self.training_stats.copy()
        
        # Add component statistics
        stats.update({
            'multi_gpu_manager': self.multi_gpu_manager.get_performance_stats(),
            'gradient_aggregator': self.gradient_aggregator.get_aggregation_stats(),
            'batch_sharder': self.batch_sharder.get_load_balance_stats(),
            'global_step': self.global_step,
            'epoch': self.epoch
        })
        
        # Calculate average time
        num_batches = max(stats['num_batches_processed'], 1)
        stats.update({
            'avg_forward_time_ms': stats['total_forward_time'] * 1000 / num_batches,
            'avg_backward_time_ms': stats['total_backward_time'] * 1000 / num_batches,
            'avg_aggregation_time_ms': stats['total_aggregation_time'] * 1000 / num_batches,
            'avg_sync_time_ms': stats['total_sync_time'] * 1000 / num_batches
        })
        
        return stats
    
    def save_checkpoint(self, checkpoint_path: str, include_optimizer: bool = True):
        """Save training checkpoint"""
        checkpoint = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'training_stats': self.training_stats,
            'trainer_config': {
                'num_gpus': self.config.num_gpus,
                'gradient_aggregation': self.config.gradient_aggregation,
                'enable_mixed_precision': self.config.enable_mixed_precision
            }
        }
        
        if include_optimizer:
            # Save optimizer state for main GPU
            self.multi_gpu_manager.save_checkpoint(checkpoint_path, include_optimizer)
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            # Load optimizer state for main GPU
            self.multi_gpu_manager.load_checkpoint(checkpoint_path, load_optimizer)
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"📂 Checkpoint loaded: {checkpoint_path}, Step={self.global_step}, Epoch={self.epoch}")
    
    def cleanup(self):
        """Clean up trainer resources"""
        logger.info("🧹 Cleaning up MultiGPUTrainer resources...")
        
        if hasattr(self, 'multi_gpu_manager'):
            self.multi_gpu_manager.cleanup()
        
        if self.scaler:
            del self.scaler
            
        # Clean up CUDA cache
        for gpu_id in range(self.config.num_gpus):
            with torch.cuda.device(gpu_id):
                torch.cuda.empty_cache()
        
        logger.info("✅ Trainer resources cleaned up")
    
    def __del__(self):
        """Destructor"""
        if hasattr(self, '_setup_complete') and self._setup_complete:
            self.cleanup()

    def _extract_optimizer_config(self, trainer: Any) -> Dict[str, Any]:
        """Extract optimizer configuration from original trainer"""
        # 1️⃣ Directly extract from already created optimizer
        if hasattr(trainer, "optimizer") and trainer.optimizer is not None:
            opt = trainer.optimizer
            base_group = opt.param_groups[0] if opt.param_groups else {}
            optimizer_kwargs = {
                'lr': base_group.get('lr', getattr(trainer.args, 'learning_rate', 5e-5)),
                'weight_decay': base_group.get('weight_decay', getattr(trainer.args, 'weight_decay', 0.0)),
                'eps': base_group.get('eps', 1e-8),
            }
            return {
                'class': opt.__class__,
                'kwargs': optimizer_kwargs,
            }

        # 2️⃣ Fallback: Use Transformers helper function
        try:
            from transformers import Trainer as HFTrainer
            optimizer_cls, optimizer_kwargs = HFTrainer.get_optimator_cls_and_kwargs(trainer.args)
            return {
                'class': optimizer_cls,
                'kwargs': optimizer_kwargs,
            }
        except Exception as e:
            logger.warning(f"Unable to extract optimizer configuration, using default AdamW: {e}")
            return {
                'class': torch.optim.AdamW,
                'kwargs': {'lr': getattr(trainer.args, 'learning_rate', 5e-5)},
            }

    def _get_aggregation_method(self, aggregation_method: str) -> AggregationMethod:
        """Convert string to AggregationMethod enum"""
        method_mapping = {
            "mean": AggregationMethod.MEAN,
            "sum": AggregationMethod.SUM,
            "weighted_mean": AggregationMethod.WEIGHTED_MEAN,
            "median": AggregationMethod.MEDIAN
        }
        
        if aggregation_method not in method_mapping:
            logger.warning(f"Unknown aggregation method: {aggregation_method}, using default value 'mean'")
            return AggregationMethod.MEAN
        
        return method_mapping[aggregation_method]

    def _accumulate_gradients(self, gradients: Dict[str, torch.Tensor]):
        """Accumulate gradients"""
        for name, grad in gradients.items():
            if name in self.accumulated_gradients:
                self.accumulated_gradients[name] += grad
            else:
                self.accumulated_gradients[name] = grad.clone().detach()

    def _apply_accumulated_gradients(self):
        """Apply and average accumulated gradients"""
        # Average accumulated gradients
        for name in self.accumulated_gradients:
            self.accumulated_gradients[name] /= self.gradient_accumulation_steps
        # Apply through MultiGPUManager on all GPUs simultaneously and execute optimizer step
        self.multi_gpu_manager.apply_aggregated_gradients(self.accumulated_gradients)

    def _apply_accumulated_gradients_old(self):
        """Apply and average accumulated gradients"""
        # Average accumulated gradients
        for name in self.accumulated_gradients:
            self.accumulated_gradients[name] /= self.gradient_accumulation_steps
        # Apply to main model
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.accumulated_gradients:
                param.grad = self.accumulated_gradients[name].to(param.device)
        # Execute optimizer step
        main_optimizer = self.multi_gpu_manager.get_optimizer(0)
        if self.scaler:
            self.scaler.step(main_optimizer)
            self.scaler.update()
        else:
            main_optimizer.step()
        main_optimizer.zero_grad()
        # Clear gradients for all GPU models
        for gpu_id in range(self.config.num_gpus):
            model = self.multi_gpu_manager.get_model(gpu_id)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad = None 