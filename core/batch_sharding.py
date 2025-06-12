"""
Batch Sharder - Load Balancing Core
Responsible for intelligent batch sharding, load balancing, and output collection
"""

import logging
from typing import Any, Dict, List, Optional, Union

import torch

logger = logging.getLogger(__name__)

class BatchSharding:
    """
    Batch Sharder - Load Balancing Core
    
    Responsible for:
    1. Intelligent batch sharding strategy
    2. Load balancing processing
    3. Processing unevenly distributed shards
    4. Output result collection
    5. Sharding correctness verification
    """
    
    def __init__(self, num_gpus: int):
        """
        Initialize sharder
        
        Args:
            num_gpus: Number of GPUs
        """
        self.num_gpus = num_gpus
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.total_batches_processed = 0
        self.total_samples_processed = 0
        self.empty_shards_count = 0
        
        logger.info(f"🔧 Initializing BatchSharding: {num_gpus} GPUs")
    
    def split_batch(self, batch: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """
        Intelligent sharding strategy
        
        Args:
            batch: Original batch data, dictionary format {'input_ids': tensor, 'labels': tensor, ...}
            
        Returns:
            List[Dict]: List of batches split for each GPU, each element corresponds to data for one GPU
        """
        if not batch:
            logger.warning("Received empty batch, returning empty shards")
            return [{}] * self.num_gpus
        
        # Get batch size (assuming all dimensions of tensors are batch dimensions)
        batch_size = self._get_batch_size(batch)
        
        if batch_size == 0:
            logger.warning("Batch size is 0, returning empty shards")
            return [{}] * self.num_gpus
        
        logger.debug(f"📦 Splitting batch: size={batch_size}, GPUs={self.num_gpus}")
        
        # Process batch when batch size is less than GPU count
        if batch_size < self.num_gpus:
            logger.warning(f"⚠️ Batch size {batch_size} < GPU count {self.num_gpus}, some GPUs will receive empty shards")
            return self._split_small_batch(batch, batch_size)
        
        # Normal batch processing
        return self._split_normal_batch(batch, batch_size)
    
    def _get_batch_size(self, batch: Dict[str, torch.Tensor]) -> int:
        """Get batch size"""
        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor) and tensor.dim() > 0:
                return tensor.size(0)
        return 0
    
    def _split_small_batch(self, batch: Dict[str, torch.Tensor], batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """Process small batch splitting (batch size < GPU count)"""
        shards = []
        
        for gpu_id in range(self.num_gpus):
            if gpu_id < batch_size:
                # Assign one sample to each GPU
                shard = {}
                for key, tensor in batch.items():
                    if isinstance(tensor, torch.Tensor):
                        shard[key] = tensor[gpu_id:gpu_id+1]  # Maintain dimension
                    else:
                        shard[key] = tensor  # Non-tensor data directly copied
                shards.append(shard)
            else:
                # Empty shard
                shard = self._create_empty_shard(batch)
                shards.append(shard)
                self.empty_shards_count += 1
        
        return shards
    
    def _split_normal_batch(self, batch: Dict[str, torch.Tensor], batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """Process normal batch splitting"""
        # Calculate each GPU's shard size
        chunk_size = batch_size // self.num_gpus
        remainder = batch_size % self.num_gpus
        
        shards = []
        start_idx = 0
        
        for gpu_id in range(self.num_gpus):
            # Remainder allocation strategy: first remainder GPUs allocate one more sample
            current_chunk_size = chunk_size + (1 if gpu_id < remainder else 0)
            end_idx = start_idx + current_chunk_size
            
            shard = {}
            for key, tensor in batch.items():
                if isinstance(tensor, torch.Tensor):
                    if start_idx < tensor.size(0):
                        shard[key] = tensor[start_idx:end_idx]
                    else:
                        # Create empty tensor to maintain shape consistency
                        empty_shape = list(tensor.shape)
                        empty_shape[0] = 0
                        shard[key] = torch.empty(empty_shape, dtype=tensor.dtype, device=tensor.device)
                else:
                    shard[key] = tensor  # Non-tensor data directly copied
            
            shards.append(shard)
            start_idx = end_idx
        
        # Verify shard correctness
        self._validate_sharding(batch, shards, batch_size)
        
        self.total_batches_processed += 1
        self.total_samples_processed += batch_size
        
        return shards
    
    def _create_empty_shard(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Create empty shard to maintain data structure consistency"""
        empty_shard = {}
        
        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor):
                # Create empty tensor to maintain all dimensions except batch dimension
                empty_shape = list(tensor.shape)
                empty_shape[0] = 0
                empty_shard[key] = torch.empty(empty_shape, dtype=tensor.dtype, device=tensor.device)
            else:
                empty_shard[key] = tensor
        
        return empty_shard
    
    def _validate_sharding(self, original_batch: Dict[str, torch.Tensor], 
                          shards: List[Dict[str, torch.Tensor]], 
                          expected_total_size: int):
        """Verify shard correctness"""
        # Check shard count
        if len(shards) != self.num_gpus:
            raise ValueError(f"Shard count mismatch: Expected {self.num_gpus}, Actual {len(shards)}")
        
        # Check total sample count
        total_samples = sum(self._get_batch_size(shard) for shard in shards)
        if total_samples != expected_total_size:
            raise ValueError(f"Total sample count mismatch after splitting: Expected {expected_total_size}, Actual {total_samples}")
        
        # Check key consistency
        original_keys = set(original_batch.keys())
        for gpu_id, shard in enumerate(shards):
            shard_keys = set(shard.keys())
            if shard_keys != original_keys:
                logger.warning(f"GPU {gpu_id} Shard key mismatch: {shard_keys} vs {original_keys}")
        
        logger.debug(f"✅ Shard verification passed: {self.num_gpus} shards, Total sample count {total_samples}")
    
    def gather_outputs(self, outputs_per_gpu: List[Any]) -> Any:
        """
        Collect outputs from each GPU
        
        Args:
            outputs_per_gpu: List of outputs from each GPU
            
        Returns:
            Aggregated output
        """
        if not outputs_per_gpu:
            return None
        
        # Filter out empty outputs
        valid_outputs = [output for output in outputs_per_gpu if output is not None]
        
        if not valid_outputs:
            return None
        
        # Aggregation strategy based on output type
        first_output = valid_outputs[0]
        
        if isinstance(first_output, (int, float)):
            # Numerical type: Calculate average
            return sum(valid_outputs) / len(valid_outputs)
        
        elif isinstance(first_output, torch.Tensor):
            # Tensor type: Concatenate along batch dimension
            if first_output.dim() == 0:
                # Scalar tensor: Average
                return torch.stack(valid_outputs).mean()
            else:
                # Multi-dimensional tensor: Concatenate
                return torch.cat(valid_outputs, dim=0)
        
        elif isinstance(first_output, dict):
            # Dictionary type: Recursive aggregation
            return self._gather_dict_outputs(valid_outputs)
        
        elif isinstance(first_output, list):
            # List type: Flatten and merge
            result = []
            for output in valid_outputs:
                result.extend(output)
            return result
        
        else:
            # Other type: Return list
            return valid_outputs
    
    def _gather_dict_outputs(self, dict_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate dictionary type outputs"""
        if not dict_outputs:
            return {}
        
        result = {}
        keys = dict_outputs[0].keys()
        
        for key in keys:
            values = [output.get(key) for output in dict_outputs if key in output]
            if values:
                result[key] = self.gather_outputs(values)
        
        return result
    
    def get_load_balance_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics"""
        avg_samples_per_batch = (self.total_samples_processed / 
                                max(self.total_batches_processed, 1))
        
        return {
            'total_batches_processed': self.total_batches_processed,
            'total_samples_processed': self.total_samples_processed,
            'empty_shards_count': self.empty_shards_count,
            'avg_samples_per_batch': avg_samples_per_batch,
            'num_gpus': self.num_gpus
        }
    
    def reset_stats(self):
        """Reset statistics"""
        self.total_batches_processed = 0
        self.total_samples_processed = 0
        self.empty_shards_count = 0
        logger.info("📊 Batch sharding statistics have been reset")


class AdaptiveBatchSharding(BatchSharding):
    """
    Adaptive Batch Sharder
    Can dynamically adjust load allocation based on GPU performance
    """
    
    def __init__(self, num_gpus: int, gpu_weights: Optional[List[float]] = None):
        """
        Initialize adaptive sharder
        
        Args:
            num_gpus: Number of GPUs
            gpu_weights: GPU weight list, used for uneven load allocation
        """
        super().__init__(num_gpus)
        
        if gpu_weights:
            if len(gpu_weights) != num_gpus:
                raise ValueError(f"GPU weight count {len(gpu_weights)} does not match GPU count {num_gpus}")
            self.gpu_weights = [w / sum(gpu_weights) for w in gpu_weights]  # Normalize
        else:
            self.gpu_weights = [1.0 / num_gpus] * num_gpus  # Uniform allocation
        
        logger.info(f"🔧 Initializing AdaptiveBatchSharding: weights={self.gpu_weights}")
    
    def _split_normal_batch(self, batch: Dict[str, torch.Tensor], batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """Adaptive splitting based on weights"""
        # Calculate sample count for each GPU based on weights
        gpu_sample_counts = []
        remaining_samples = batch_size
        
        for i, weight in enumerate(self.gpu_weights[:-1]):
            samples = int(batch_size * weight)
            gpu_sample_counts.append(samples)
            remaining_samples -= samples
        
        # Last GPU allocates remaining samples
        gpu_sample_counts.append(remaining_samples)
        
        # Create shards
        shards = []
        start_idx = 0
        
        for gpu_id, sample_count in enumerate(gpu_sample_counts):
            end_idx = start_idx + sample_count
            
            shard = {}
            for key, tensor in batch.items():
                if isinstance(tensor, torch.Tensor):
                    if start_idx < tensor.size(0):
                        shard[key] = tensor[start_idx:end_idx]
                    else:
                        empty_shape = list(tensor.shape)
                        empty_shape[0] = 0
                        shard[key] = torch.empty(empty_shape, dtype=tensor.dtype, device=tensor.device)
                else:
                    shard[key] = tensor
            
            shards.append(shard)
            start_idx = end_idx
        
        logger.debug(f"📊 Adaptive splitting: {[len(shard.get('input_ids', [])) for shard in shards]}")
        
        return shards 