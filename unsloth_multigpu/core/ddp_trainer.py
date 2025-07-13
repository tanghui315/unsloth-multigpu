"""
DDP Trainer - 基于PyTorch DDP的分布式训练器
替换原有的低效串行训练实现
"""

import logging
import time
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from transformers import Trainer

from .ddp_manager import DDPManager
from ..utils import MultiGPUConfig

logger = logging.getLogger(__name__)


class DDPTrainer:
    """
    DDP Trainer - 基于PyTorch原生DDP的高效分布式训练器
    
    主要优势：
    1. 真正的并行训练（vs 原来的串行）
    2. 自动梯度同步（vs 手动CPU传输）
    3. 高效的NCCL通信（vs 低效的聚合）
    4. 原生PyTorch支持（vs 自制轮子）
    """
    
    def __init__(self, 
                 original_trainer: Trainer,
                 config: MultiGPUConfig,
                 ddp_manager: Optional[DDPManager] = None):
        """
        初始化DDP训练器
        
        Args:
            original_trainer: 原始HuggingFace Trainer
            config: 多GPU配置
            ddp_manager: DDP管理器，None表示自动创建
        """
        self.original_trainer = original_trainer
        self.config = config
        self.ddp_manager = ddp_manager or DDPManager()
        
        # 模型和优化器
        self.model = original_trainer.model
        self.ddp_model = None
        self.optimizer = None
        
        # 数据相关
        self.train_dataset = original_trainer.train_dataset
        self.data_collator = original_trainer.data_collator
        self.train_dataloader = None
        
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.is_setup = False
        
        # 性能统计
        self.stats = {
            'total_train_time': 0.0,
            'total_forward_time': 0.0,
            'total_backward_time': 0.0,
            'total_samples': 0,
            'steps_per_second': 0.0
        }
        
        logger.info(f"🔧 初始化DDPTrainer: {config.num_gpus} GPUs")
    
    def setup(self, rank: int):
        """
        设置DDP训练环境
        
        Args:
            rank: 当前进程rank
        """
        if self.is_setup:
            return
            
        logger.info(f"🚀 设置DDP训练环境 (rank={rank})...")
        
        # 1. 初始化DDP进程组
        success = self.ddp_manager.init_process_group(
            rank=rank,
            world_size=self.config.num_gpus
        )
        if not success:
            raise RuntimeError(f"DDP进程组初始化失败 (rank={rank})")
        
        # 2. 包装模型为DDP
        self.ddp_model = self.ddp_manager.wrap_model(
            self.model,
            find_unused_parameters=getattr(self.config, 'find_unused_parameters', False)
        )
        
        # 3. 创建优化器（必须在DDP包装后）
        self._setup_optimizer()
        
        # 4. 创建分布式数据加载器
        self._setup_dataloader()
        
        self.is_setup = True
        
        if self.ddp_manager.is_master:
            logger.info("✅ DDP训练环境设置完成")
    
    def _setup_optimizer(self):
        """设置优化器"""
        # 从原始trainer提取优化器配置
        training_args = self.original_trainer.args
        
        # 创建优化器
        if hasattr(self.original_trainer, 'create_optimizer'):
            # 使用HuggingFace的优化器创建方法
            self.original_trainer.model = self.ddp_model
            self.original_trainer.create_optimizer()
            self.optimizer = self.original_trainer.optimizer
        else:
            # fallback到默认AdamW
            self.optimizer = torch.optim.AdamW(
                self.ddp_model.parameters(),
                lr=training_args.learning_rate,
                weight_decay=training_args.weight_decay,
                eps=training_args.adam_epsilon
            )
        
        if self.ddp_manager.is_master:
            logger.info(f"✅ 优化器设置完成: {type(self.optimizer).__name__}")
    
    def _setup_dataloader(self):
        """设置分布式数据加载器"""
        # 创建分布式采样器
        sampler = self.ddp_manager.create_data_sampler(
            self.train_dataset,
            shuffle=True
        )
        
        # 创建DataLoader
        self.train_dataloader = DataLoader(
            self.train_dataset,
            sampler=sampler,
            batch_size=self.config.batch_size_per_gpu,
            collate_fn=self.data_collator,
            num_workers=getattr(self.config, 'dataloader_num_workers', 0),
            pin_memory=True
        )
        
        if self.ddp_manager.is_master:
            logger.info(f"✅ 分布式数据加载器设置完成: batch_size={self.config.batch_size_per_gpu}")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        执行单步训练
        
        Args:
            batch: 训练批次数据
            
        Returns:
            Dict: 训练结果统计
        """
        step_start = time.time()
        
        # 移动数据到GPU
        device = f"cuda:{self.ddp_manager.local_rank}"
        batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # 梯度清零
        self.optimizer.zero_grad()
        
        # 前向传播
        forward_start = time.time()
        
        # 支持混合精度训练
        if getattr(self.config, 'fp16', False):
            with torch.cuda.amp.autocast():
                outputs = self.ddp_model(**batch)
                loss = outputs.loss
        else:
            outputs = self.ddp_model(**batch)
            loss = outputs.loss
        
        forward_time = time.time() - forward_start
        
        # 反向传播（DDP自动同步梯度）
        backward_start = time.time()
        loss.backward()
        backward_time = time.time() - backward_start
        
        # 梯度裁剪
        if getattr(self.config, 'max_grad_norm', None):
            torch.nn.utils.clip_grad_norm_(
                self.ddp_model.parameters(), 
                self.config.max_grad_norm
            )
        
        # 优化器步骤
        self.optimizer.step()
        
        # 更新全局步数
        self.global_step += 1
        
        # 统计信息
        step_time = time.time() - step_start
        batch_size = batch['input_ids'].size(0)
        
        self.stats['total_forward_time'] += forward_time
        self.stats['total_backward_time'] += backward_time
        self.stats['total_samples'] += batch_size
        
        return {
            'loss': loss.item(),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'step_time': step_time,
            'forward_time': forward_time,
            'backward_time': backward_time,
            'throughput': batch_size / step_time,
            'global_step': self.global_step
        }
    
    def train_epoch(self) -> Dict[str, Any]:
        """
        训练一个epoch
        
        Returns:
            Dict: epoch训练统计
        """
        if not self.is_setup:
            raise RuntimeError("DDP训练器未设置，请先调用setup()")
        
        epoch_start = time.time()
        epoch_loss = 0.0
        num_steps = 0
        
        # 设置采样器的epoch（确保每个epoch数据顺序不同）
        if hasattr(self.train_dataloader.sampler, 'set_epoch'):
            self.train_dataloader.sampler.set_epoch(self.epoch)
        
        self.ddp_model.train()
        
        if self.ddp_manager.is_master:
            logger.info(f"🚀 开始训练 epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            step_result = self.train_step(batch)
            epoch_loss += step_result['loss']
            num_steps += 1
            
            # 定期打印进度（仅主进程）
            if self.ddp_manager.is_master and batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {self.epoch}, Step {batch_idx}/{len(self.train_dataloader)}, "
                    f"Loss: {step_result['loss']:.4f}, "
                    f"LR: {step_result['learning_rate']:.2e}, "
                    f"Throughput: {step_result['throughput']:.1f} samples/s"
                )
        
        # Epoch统计
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / num_steps if num_steps > 0 else 0.0
        
        self.epoch += 1
        self.stats['total_train_time'] += epoch_time
        self.stats['steps_per_second'] = num_steps / epoch_time
        
        epoch_stats = {
            'epoch': self.epoch - 1,
            'avg_loss': avg_loss,
            'epoch_time': epoch_time,
            'num_steps': num_steps,
            'steps_per_second': self.stats['steps_per_second']
        }
        
        if self.ddp_manager.is_master:
            logger.info(f"✅ Epoch {self.epoch-1} 完成: "
                       f"avg_loss={avg_loss:.4f}, "
                       f"time={epoch_time:.1f}s, "
                       f"steps/s={epoch_stats['steps_per_second']:.1f}")
        
        return epoch_stats
    
    def train(self, num_epochs: Optional[int] = None) -> Dict[str, Any]:
        """
        执行完整训练
        
        Args:
            num_epochs: 训练轮数，None表示使用配置中的值
            
        Returns:
            Dict: 训练结果统计
        """
        if num_epochs is None:
            num_epochs = getattr(self.config, 'num_train_epochs', 3)
        
        train_start = time.time()
        all_epoch_stats = []
        
        if self.ddp_manager.is_master:
            logger.info(f"🚀 开始DDP训练: {num_epochs} epochs, {self.config.num_gpus} GPUs")
        
        try:
            for epoch_idx in range(num_epochs):
                epoch_stats = self.train_epoch()
                all_epoch_stats.append(epoch_stats)
            
            # 最终统计
            total_time = time.time() - train_start
            final_stats = {
                'total_time': total_time,
                'num_epochs': num_epochs,
                'total_steps': self.global_step,
                'avg_steps_per_second': self.global_step / total_time,
                'total_samples': self.stats['total_samples'],
                'epoch_stats': all_epoch_stats,
                'performance_stats': self.stats
            }
            
            if self.ddp_manager.is_master:
                logger.info(f"🎉 DDP训练完成! "
                           f"总时间: {total_time:.1f}s, "
                           f"总步数: {self.global_step}, "
                           f"平均速度: {final_stats['avg_steps_per_second']:.1f} steps/s")
            
            return final_stats
            
        except Exception as e:
            logger.error(f"❌ DDP训练过程中出错: {e}")
            raise
        finally:
            # 清理资源
            self.cleanup()
    
    def cleanup(self):
        """清理DDP资源"""
        try:
            self.ddp_manager.cleanup()
            if self.ddp_manager.is_master:
                logger.info("✅ DDP资源清理完成")
        except Exception as e:
            logger.warning(f"⚠️ DDP清理过程中出现警告: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        return {
            'ddp_stats': self.ddp_manager.get_status(),
            'training_stats': self.stats.copy(),
            'config': {
                'num_gpus': self.config.num_gpus,
                'batch_size_per_gpu': self.config.batch_size_per_gpu,
                'gradient_aggregation': self.config.gradient_aggregation
            }
        }
    
    def __del__(self):
        """析构函数"""
        try:
            self.cleanup()
        except:
            pass