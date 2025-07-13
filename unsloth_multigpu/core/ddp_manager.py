"""
DDP Manager - PyTorch原生分布式训练管理器
基于PyTorch DistributedDataParallel实现真正的并行训练
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
    DDP Manager - 基于PyTorch原生DDP的分布式训练管理器
    
    主要功能：
    1. 分布式进程组初始化
    2. DDP模型包装和管理
    3. 进程间通信协调
    4. 错误处理和资源清理
    """
    
    def __init__(self, backend: str = "nccl", timeout_minutes: int = 30):
        """
        初始化DDP管理器
        
        Args:
            backend: 分布式后端，GPU推荐nccl，CPU推荐gloo
            timeout_minutes: 分布式操作超时时间
        """
        self.backend = backend
        self.timeout = timeout_minutes * 60  # 转换为秒
        
        # 分布式状态
        self.is_initialized = False
        self.rank = None
        self.local_rank = None
        self.world_size = None
        self.is_master = False
        
        # DDP模型实例
        self.ddp_model = None
        self.original_model = None
        
        logger.info(f"🔧 初始化DDPManager: backend={backend}, timeout={timeout_minutes}min")
    
    def init_process_group(self, rank: int, world_size: int, 
                          master_addr: str = "localhost", 
                          master_port: str = "12355") -> bool:
        """
        初始化分布式进程组
        
        Args:
            rank: 当前进程在全局的排名
            world_size: 总进程数
            master_addr: 主节点地址
            master_port: 主节点端口
            
        Returns:
            bool: 是否成功初始化
        """
        try:
            # 设置环境变量
            os.environ['MASTER_ADDR'] = master_addr
            os.environ['MASTER_PORT'] = master_port
            os.environ['WORLD_SIZE'] = str(world_size)
            os.environ['RANK'] = str(rank)
            
            # 计算local_rank（单机多GPU场景）
            local_rank = rank % torch.cuda.device_count()
            os.environ['LOCAL_RANK'] = str(local_rank)
            
            # 初始化进程组
            dist.init_process_group(
                backend=self.backend,
                rank=rank,
                world_size=world_size,
                timeout=torch.distributed.default_pg_timeout if self.timeout == 1800 else 
                       torch.distributed.init_process_group.__defaults__[3]
            )
            
            # 设置CUDA设备
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
            
            # 更新状态
            self.rank = rank
            self.local_rank = local_rank
            self.world_size = world_size
            self.is_master = (rank == 0)
            self.is_initialized = True
            
            if self.is_master:
                logger.info(f"✅ DDP进程组初始化成功: rank={rank}, world_size={world_size}, backend={self.backend}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ DDP进程组初始化失败: {e}")
            return False
    
    def wrap_model(self, model: torch.nn.Module, 
                   device_ids: Optional[list] = None,
                   find_unused_parameters: bool = False,
                   broadcast_buffers: bool = True) -> torch.nn.Module:
        """
        用DDP包装模型
        
        Args:
            model: 要包装的模型
            device_ids: 设备ID列表，None表示自动推断
            find_unused_parameters: 是否查找未使用的参数
            broadcast_buffers: 是否广播buffer
            
        Returns:
            torch.nn.Module: DDP包装后的模型
        """
        if not self.is_initialized:
            raise RuntimeError("DDP进程组未初始化，请先调用init_process_group()")
        
        # 移动模型到当前GPU
        if torch.cuda.is_available():
            device = f"cuda:{self.local_rank}"
            model = model.to(device)
        
        # 自动推断device_ids
        if device_ids is None and torch.cuda.is_available():
            device_ids = [self.local_rank]
        
        # 包装模型
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
                logger.info(f"✅ 模型DDP包装完成: {param_count:,} 参数")
            
            return self.ddp_model
            
        except Exception as e:
            logger.error(f"❌ 模型DDP包装失败: {e}")
            raise
    
    def create_data_sampler(self, dataset, shuffle: bool = True):
        """
        创建分布式数据采样器
        
        Args:
            dataset: 数据集
            shuffle: 是否打乱数据
            
        Returns:
            DistributedSampler: 分布式采样器
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
        暂停梯度同步的上下文管理器
        用于梯度累积场景
        """
        if self.ddp_model is not None:
            with self.ddp_model.no_sync():
                yield
        else:
            yield
    
    def barrier(self):
        """进程同步屏障"""
        if self.is_initialized:
            dist.barrier()
    
    def all_reduce(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM):
        """
        跨进程tensor归约操作
        
        Args:
            tensor: 要归约的tensor
            op: 归约操作类型
        """
        if self.is_initialized:
            dist.all_reduce(tensor, op=op)
        return tensor
    
    def broadcast(self, tensor: torch.Tensor, src: int = 0):
        """
        从源进程广播tensor到所有进程
        
        Args:
            tensor: 要广播的tensor
            src: 源进程rank
        """
        if self.is_initialized:
            dist.broadcast(tensor, src=src)
        return tensor
    
    def cleanup(self):
        """清理分布式资源"""
        try:
            if self.is_initialized:
                dist.destroy_process_group()
                self.is_initialized = False
                
                if self.is_master:
                    logger.info("✅ DDP进程组已清理")
                    
        except Exception as e:
            logger.warning(f"⚠️ DDP清理过程中出现警告: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取DDP状态信息"""
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
        """析构函数，确保资源被清理"""
        if self.is_initialized:
            self.cleanup()


def find_free_port() -> str:
    """寻找空闲端口"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return str(s.getsockname()[1])


def setup_ddp_environment(num_gpus: int, 
                         master_addr: str = "localhost",
                         master_port: Optional[str] = None) -> Dict[str, str]:
    """
    设置DDP环境变量
    
    Args:
        num_gpus: GPU数量
        master_addr: 主节点地址
        master_port: 主节点端口，None表示自动寻找
        
    Returns:
        Dict: 环境变量字典
    """
    if master_port is None:
        master_port = find_free_port()
    
    env_vars = {
        'MASTER_ADDR': master_addr,
        'MASTER_PORT': master_port,
        'WORLD_SIZE': str(num_gpus),
        'NCCL_DEBUG': 'INFO',  # 调试信息
        'CUDA_VISIBLE_DEVICES': ','.join(str(i) for i in range(num_gpus))
    }
    
    # 设置环境变量
    for key, value in env_vars.items():
        os.environ[key] = value
    
    logger.info(f"🔧 DDP环境变量设置完成: {env_vars}")
    return env_vars