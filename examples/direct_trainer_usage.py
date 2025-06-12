"""
Direct usage example of MultiGPUTrainer
Demonstrates how to bypass the Hook mechanism and use our Multi-GPU trainer directly
"""

import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import TrainingArguments

from unsloth import FastLanguageModel
# 直接导入我们的核心组件
from unsloth_multigpu_prototype.core import AggregationMethod, MultiGPUTrainer
from unsloth_multigpu_prototype.utils import (ConfigManager, DeviceManager,
                                              MultiGPULogger)


def create_dataloader(dataset, tokenizer, batch_size=8):
    """创建数据加载器"""
    def tokenize_function(examples):
        # 简化的分词函数
        inputs = tokenizer(
            examples['instruction'], 
            truncation=True, 
            padding=True, 
            max_length=512,
            return_tensors="pt"
        )
        # 添加标签（在实际使用中，您可能需要更复杂的处理）
        inputs['labels'] = inputs['input_ids'].clone()
        return inputs
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    return DataLoader(
        tokenized_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=lambda x: {k: torch.stack([item[k] for item in x]) for k in x[0]}
    )

def main():
    """直接使用MultiGPUTrainer的主函数"""
    
    # 1. 初始化设备和配置管理
    print("🔧 初始化配置...")
    device_manager = DeviceManager()
    config_manager = ConfigManager()
    logger = MultiGPULogger(log_dir="./logs/direct_trainer")
    
    # 检查可用设备
    available_devices = device_manager.get_available_devices()
    num_gpus = min(len(available_devices), 4)  # 最多使用4个GPU
    
    if num_gpus < 2:
        print("⚠️ 多GPU训练需要至少2个GPU，当前环境只能进行演示")
        num_gpus = 1  # 降级到单GPU模式
    
    print(f"📊 使用 {num_gpus} 个GPU进行训练")
    
    # 2. 加载模型
    print("📥 加载模型...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        "unsloth/llama-2-7b-bnb-4bit",
        max_seq_length=2048,
        dtype="bfloat16",
        load_in_4bit=True
    )
    
    # 3. 准备数据
    print("📚 准备数据集...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train[:100]")  # 使用小数据集演示
    
    # 创建数据加载器
    dataloader = create_dataloader(dataset, tokenizer, batch_size=4)
    
    # 4. 配置优化器
    optimizer_config = {
        'class': torch.optim.AdamW,
        'kwargs': {
            'lr': 2e-5,
            'weight_decay': 0.01,
            'betas': (0.9, 0.999)
        }
    }
    
    # 5. 创建MultiGPUTrainer
    print("🚀 初始化MultiGPUTrainer...")
    trainer = MultiGPUTrainer(
        model=model,
        num_gpus=num_gpus,
        optimizer_config=optimizer_config,
        aggregation_method=AggregationMethod.MEAN,
        enable_gradient_checkpointing=True,
        enable_mixed_precision=True
    )
    
    # 6. 设置训练器
    trainer.setup()
    
    # 7. 开始训练
    print("🎯 开始训练...")
    
    try:
        # 训练多个epoch
        for epoch in range(2):  # 训练2个epoch演示
            print(f"\n📖 Epoch {epoch + 1}/2")
            
            epoch_stats = trainer.train_epoch(
                dataloader=dataloader,
                max_steps=10  # 限制步数用于演示
            )
            
            print(f"✅ Epoch {epoch + 1} 完成:")
            print(f"  - 平均损失: {epoch_stats['avg_loss']:.4f}")
            print(f"  - 训练步数: {epoch_stats['steps']}")
            print(f"  - 处理样本: {epoch_stats['samples']}")
            print(f"  - 耗时: {epoch_stats['epoch_time']:.2f}s")
        
        # 8. 获取训练统计
        print("\n📊 训练统计:")
        training_stats = trainer.get_training_stats()
        
        print(f"  - 总步数: {training_stats['global_step']}")
        print(f"  - 处理批次: {training_stats['num_batches_processed']}")
        print(f"  - 处理样本: {training_stats['num_samples_processed']}")
        print(f"  - 平均前向时间: {training_stats.get('avg_forward_time_ms', 0):.2f}ms")
        print(f"  - 平均反向时间: {training_stats.get('avg_backward_time_ms', 0):.2f}ms")
        print(f"  - 平均聚合时间: {training_stats.get('avg_aggregation_time_ms', 0):.2f}ms")
        
        # 9. 保存检查点
        checkpoint_path = "./checkpoints/direct_trainer_checkpoint.pt"
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        trainer.save_checkpoint(checkpoint_path)
        print(f"💾 检查点已保存: {checkpoint_path}")
        
    except Exception as e:
        print(f"❌ 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 10. 清理资源
        print("🧹 清理资源...")
        trainer.cleanup()
        print("✅ 直接使用MultiGPUTrainer示例完成!")

def compare_approaches():
    """比较两种使用方式"""
    print("\n" + "="*60)
    print("📋 两种使用方式对比")
    print("="*60)
    
    print("\n🔧 方式1: Hook机制 (quick_start.py)")
    print("  优点:")
    print("    ✅ 零侵入性 - 现有代码无需修改")
    print("    ✅ 向后兼容 - 完全兼容Unsloth API")
    print("    ✅ 简单易用 - 只需enable_multi_gpu()")
    print("  缺点:")
    print("    ⚠️ 不够直观 - 用户可能不知道内部机制")
    print("    ⚠️ 调试复杂 - Hook替换可能难以调试")
    
    print("\n🎯 方式2: 直接使用 (本示例)")
    print("  优点:")
    print("    ✅ 完全控制 - 用户明确知道使用的是什么")
    print("    ✅ 易于调试 - 直接访问所有组件")
    print("    ✅ 灵活配置 - 可以精细调整所有参数")
    print("  缺点:")
    print("    ⚠️ 需要学习 - 用户需要了解新API")
    print("    ⚠️ 代码修改 - 需要修改现有训练代码")
    
    print("\n💡 建议:")
    print("  - 快速迁移: 使用Hook机制")
    print("  - 新项目/高级用户: 直接使用MultiGPUTrainer")
    print("  - 调试/定制: 直接使用MultiGPUTrainer")

if __name__ == "__main__":
    main()
    compare_approaches() 