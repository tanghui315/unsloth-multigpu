"""
DDP Launcher - 分布式训练启动器
基于torch.multiprocessing实现多进程DDP训练
"""

import logging
import os
import sys
from typing import Any, Callable, Dict, Optional

import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Process

from .ddp_manager import setup_ddp_environment
from .ddp_trainer import DDPTrainer
from ..utils import MultiGPUConfig

logger = logging.getLogger(__name__)


class DDPLauncher:
    """
    DDP Launcher - 分布式训练启动器
    
    主要功能：
    1. 管理多进程训练启动
    2. 错误处理和进程监控
    3. 结果收集和汇总
    4. 资源清理
    """
    
    def __init__(self, config: MultiGPUConfig):
        """
        初始化DDP启动器
        
        Args:
            config: 多GPU配置
        """
        self.config = config
        self.processes = []
        self.results = {}
        
        logger.info(f"🔧 初始化DDPLauncher: {config.num_gpus} GPUs")
    
    def launch_training(self, 
                       trainer_fn: Callable,
                       trainer_args: tuple = (),
                       trainer_kwargs: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        启动分布式训练
        
        Args:
            trainer_fn: 训练函数
            trainer_args: 训练函数位置参数
            trainer_kwargs: 训练函数关键字参数
            
        Returns:
            Dict: 训练结果
        """
        if trainer_kwargs is None:
            trainer_kwargs = {}
            
        logger.info(f"🚀 启动DDP训练: {self.config.num_gpus} 进程")
        
        try:
            # 设置DDP环境
            setup_ddp_environment(self.config.num_gpus)
            
            # 使用spawn方法启动进程（推荐用于CUDA）
            mp.set_start_method('spawn', force=True)
            
            # 创建进程
            processes = []
            for rank in range(self.config.num_gpus):
                p = Process(
                    target=self._worker_process,
                    args=(rank, trainer_fn, trainer_args, trainer_kwargs)
                )
                p.start()
                processes.append(p)
            
            # 等待所有进程完成
            for p in processes:
                p.join()
            
            # 检查进程状态
            success_count = 0
            for rank, p in enumerate(processes):
                if p.exitcode == 0:
                    success_count += 1
                else:
                    logger.error(f"❌ 进程 {rank} 退出异常: exit_code={p.exitcode}")
            
            if success_count == self.config.num_gpus:
                logger.info("✅ 所有DDP进程成功完成")
                return {'status': 'success', 'num_processes': success_count}
            else:
                logger.error(f"❌ DDP训练失败: {success_count}/{self.config.num_gpus} 进程成功")
                return {'status': 'failed', 'success_count': success_count}
                
        except Exception as e:
            logger.error(f"❌ DDP启动失败: {e}")
            self._cleanup_processes()
            raise
    
    def _worker_process(self, 
                       rank: int, 
                       trainer_fn: Callable,
                       trainer_args: tuple,
                       trainer_kwargs: Dict[str, Any]):
        """
        工作进程函数
        
        Args:
            rank: 进程rank
            trainer_fn: 训练函数
            trainer_args: 训练函数参数
            trainer_kwargs: 训练函数关键字参数
        """
        try:
            # 设置进程日志
            process_logger = logging.getLogger(f"ddp_worker_{rank}")
            process_logger.info(f"🚀 启动DDP工作进程 (rank={rank})")
            
            # 调用训练函数
            result = trainer_fn(rank, *trainer_args, **trainer_kwargs)
            
            process_logger.info(f"✅ DDP工作进程完成 (rank={rank})")
            return result
            
        except Exception as e:
            logger.error(f"❌ DDP工作进程失败 (rank={rank}): {e}")
            logger.error(f"异常详情:", exc_info=True)
            sys.exit(1)
    
    def _cleanup_processes(self):
        """清理进程资源"""
        for p in self.processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5.0)
                if p.is_alive():
                    p.kill()
        
        self.processes.clear()
        logger.info("🧹 进程资源清理完成")


def ddp_train_worker(rank: int, 
                    original_trainer,
                    config: MultiGPUConfig) -> Dict[str, Any]:
    """
    DDP训练工作函数
    
    Args:
        rank: 进程rank
        original_trainer: 原始trainer
        config: 多GPU配置
        
    Returns:
        Dict: 训练结果
    """
    from .ddp_manager import DDPManager
    
    try:
        # 创建DDP管理器和训练器
        ddp_manager = DDPManager()
        ddp_trainer = DDPTrainer(original_trainer, config, ddp_manager)
        
        # 设置DDP环境
        ddp_trainer.setup(rank)
        
        # 执行训练
        result = ddp_trainer.train()
        
        return result
        
    except Exception as e:
        logger.error(f"❌ DDP训练工作进程失败 (rank={rank}): {e}")
        raise


def launch_ddp_training(original_trainer, config: MultiGPUConfig) -> Dict[str, Any]:
    """
    便捷的DDP训练启动函数
    
    Args:
        original_trainer: 原始HuggingFace trainer
        config: 多GPU配置
        
    Returns:
        Dict: 训练结果
    """
    launcher = DDPLauncher(config)
    
    return launcher.launch_training(
        trainer_fn=ddp_train_worker,
        trainer_args=(original_trainer, config)
    )