"""
DDP Manager - Native PyTorch Distributed Training Manager
Based on PyTorch DistributedDataParallel for true parallel training
"""

import logging
import os
import socket
from contextlib import contextmanager
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)


class DDPManager:
    """
    DDP Manager - Distributed training manager based on native PyTorch DDP
    
    Main features:
    1. Distributed process group initialization
    2. DDP model wrapping and management
    3. Inter-process communication coordination
    4. Error handling and resource cleanup
    """
    
    def __init__(self, backend: str = "nccl", timeout_minutes: int = 30):
        """
        Initialize DDP manager
        
        Args:
            backend: Distributed backend, recommend 'nccl' for GPU, 'gloo' for CPU
            timeout_minutes: Timeout for distributed operations
        """
        self.backend = backend
        self.timeout = timeout_minutes * 60  # Convert to seconds
        
        # Distributed state
        self.is_initialized = False
        self.rank = None
        self.local_rank = None
        self.world_size = None
        self.is_master = False
        
        # DDP model instance
        self.ddp_model = None
        self.original_model = None
        
        logger.info(f"🔧 Initializing DDPManager: backend={backend}, timeout={timeout_minutes}min")
    
    def init_process_group(self, rank: int, world_size: int, 
                          master_addr: str = "localhost", 
                          master_port: str = "12355") -> bool:
        """
        Initialize distributed process group
        
        Args:
            rank: Global rank of current process
            world_size: Total number of processes
            master_addr: Master node address
            master_port: Master node port
            
        Returns:
            bool: Whether initialization succeeded
        """
        try:
            # Set environment variables
            os.environ['MASTER_ADDR'] = master_addr
            os.environ['MASTER_PORT'] = master_port
            os.environ['WORLD_SIZE'] = str(world_size)
            os.environ['RANK'] = str(rank)
            
            # Calculate local_rank (for single machine multi-GPU)
            local_rank = rank % torch.cuda.device_count()
            os.environ['LOCAL_RANK'] = str(local_rank)
            
            # Initialize process group
            dist.init_process_group(
                backend=self.backend,
                rank=rank,
                world_size=world_size,
                timeout=torch.distributed.default_pg_timeout if self.timeout == 1800 else 
                       torch.distributed.init_process_group.__defaults__[3]
            )
            
            # Set CUDA device
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
            
            # Update state
            self.rank = rank
            self.local_rank = local_rank
            self.world_size = world_size
            self.is_master = (rank == 0)
            self.is_initialized = True
            
            if self.is_master:
                logger.info(f"✅ DDP process group initialized: rank={rank}, world_size={world_size}, backend={self.backend}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ DDP process group initialization failed: {e}")
            return False
    
    def wrap_model(self, model: torch.nn.Module, 
                   device_ids: Optional[list] = None,
                   find_unused_parameters: bool = False,
                   broadcast_buffers: bool = True) -> torch.nn.Module:
        """
        Wrap model with DDP
        
        Args:
            model: Model to wrap
            device_ids: List of device IDs, None for auto inference
            find_unused_parameters: Whether to find unused parameters
            broadcast_buffers: Whether to broadcast buffers
            
        Returns:
            torch.nn.Module: DDP-wrapped model
        """
        if not self.is_initialized:
            raise RuntimeError("DDP process group not initialized, please call init_process_group() first")
        
        # Move model to current GPU
        if torch.cuda.is_available():
            device = f"cuda:{self.local_rank}"
            model = model.to(device)
        
        # Auto infer device_ids
        if device_ids is None and torch.cuda.is_available():
            device_ids = [self.local_rank]
        
        # Wrap model
        try:
            self.original_model = model
            self.ddp_model = DDP(
                model,
                device_ids=device_ids,
                find_unused_parameters=find_unused_parameters,
                broadcast_buffers=broadcast_buffers
            )
            
            if self.is_master:
                param_count = sum(p.numel() for p in model.parameters())
                logger.info(f"✅ Model wrapped with DDP: {param_count:,} parameters")
            
            return self.ddp_model
            
        except Exception as e:
            logger.error(f"❌ Model DDP wrapping failed: {e}")
            raise
    
    def create_data_sampler(self, dataset, shuffle: bool = True):
        """
        Create distributed data sampler
        
        Args:
            dataset: Dataset
            shuffle: Whether to shuffle data
            
        Returns:
            DistributedSampler: Distributed data sampler
        """
        from torch.utils.data.distributed import DistributedSampler
        
        return DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle
        )
    
    @contextmanager
    def no_sync(self):
        """
        Context manager to pause gradient synchronization
        Used for gradient accumulation
        """
        if self.ddp_model is not None:
            with self.ddp_model.no_sync():
                yield
        else:
            yield
    
    def barrier(self):
        """Process synchronization barrier"""
        if self.is_initialized:
            dist.barrier()
    
    def all_reduce(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM):
        """
        All-reduce operation across processes
        
        Args:
            tensor: Tensor to reduce
            op: Reduction operation type
        """
        if self.is_initialized:
            dist.all_reduce(tensor, op=op)
        return tensor
    
    def broadcast(self, tensor: torch.Tensor, src: int = 0):
        """
        Broadcast tensor from source process to all processes
        
        Args:
            tensor: Tensor to broadcast
            src: Source process rank
        """
        if self.is_initialized:
            dist.broadcast(tensor, src=src)
        return tensor
    
    def cleanup(self):
        """Cleanup distributed resources"""
        try:
            if self.is_initialized:
                dist.destroy_process_group()
                self.is_initialized = False
                
                if self.is_master:
                    logger.info("✅ DDP process group cleaned up")
                    
        except Exception as e:
            logger.warning(f"⚠️ Warning during DDP cleanup: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get DDP status information"""
        return {
            'initialized': self.is_initialized,
            'backend': self.backend,
            'rank': self.rank,
            'local_rank': self.local_rank,
            'world_size': self.world_size,
            'is_master': self.is_master,
            'has_ddp_model': self.ddp_model is not None,
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    
    def __del__(self):
        """Destructor to ensure resources are cleaned up"""
        if self.is_initialized:
            self.cleanup()


def find_free_port() -> str:
    """Find a free port"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return str(s.getsockname()[1])


def setup_ddp_environment(num_gpus: int, 
                         master_addr: str = "localhost",
                         master_port: Optional[str] = None) -> Dict[str, str]:
    """
    Set DDP environment variables
    
    Args:
        num_gpus: Number of GPUs
        master_addr: Master node address
        master_port: Master node port, None for auto finding
        
    Returns:
        Dict: Environment variable dictionary
    """
    if master_port is None:
        master_port = find_free_port()
    
    env_vars = {
        'MASTER_ADDR': master_addr,
        'MASTER_PORT': master_port,
        'WORLD_SIZE': str(num_gpus),
        'NCCL_DEBUG': 'INFO',  # Debug info
        'CUDA_VISIBLE_DEVICES': ','.join(str(i) for i in range(num_gpus))
    }
    
    # Set environment variables
    for key, value in env_vars.items():
        os.environ[key] = value
    
    logger.info(f"🔧 DDP environment variables set: {env_vars}")
    return env_vars