"""
Unsloth Multi-GPU Advanced Configuration Example
This example demonstrates how to use Unsloth Multi-GPU's advanced features and configuration options
"""

import os

import torch
from datasets import load_dataset
from transformers import TrainingArguments

import unsloth_multigpu_prototype as unsloth_multigpu
from unsloth import FastLanguageModel, unsloth_train
from unsloth_multigpu_prototype.utils import (ConfigManager, DeviceManager,
                                              MultiGPULogger)


def main():
    # 1. 初始化日志系统
    logger = MultiGPULogger(
        log_dir="./logs",
        log_level="INFO",
        enable_tensorboard=True
    )
    logger.info("🚀 开始Unsloth多GPU高级配置示例")

    # 2. 设备管理
    device_manager = DeviceManager()
    devices = device_manager.get_available_devices()
    logger.info(f"📊 可用GPU: {devices}")

    # 3. 配置管理
    config_manager = ConfigManager()
    optimal_config = config_manager.get_optimal_config(
        model_size="7B",
        available_memory="32GB",
        num_gpus=len(devices)
    )
    logger.info(f"⚙️ 推荐配置: {optimal_config}")

    # 4. 启用多GPU（使用优化配置）
    logger.info("🔄 启用多GPU支持...")
    unsloth_multigpu.enable_multi_gpu(
        **optimal_config,
        gradient_aggregation="weighted_mean",  # 使用加权平均聚合
        memory_efficient=True,  # 启用内存优化
        enable_profiling=True   # 启用性能分析
    )

    # 5. 检查系统状态
    status = unsloth_multigpu.get_multi_gpu_status()
    logger.info(f"📊 多GPU状态: {status}")

    # 6. 加载模型
    logger.info("📥 加载模型...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        "unsloth/llama-2-7b-bnb-4bit",
        max_seq_length=2048,
        dtype="bfloat16",
        load_in_4bit=True
    )

    # 7. 准备数据集
    logger.info("📚 准备数据集...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    
    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['instruction'])):
            text = f"### Instruction:\n{example['instruction'][i]}\n\n"
            if example['input'][i]:
                text += f"### Input:\n{example['input'][i]}\n\n"
            text += f"### Response:\n{example['output'][i]}"
            output_texts.append(text)
        return output_texts

    # 8. 配置训练参数
    logger.info("⚙️ 配置训练参数...")
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=optimal_config["batch_size"],
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        warmup_ratio=0.1,
        # 高级训练配置
        gradient_checkpointing=True,  # 启用梯度检查点
        optim="adamw_torch",         # 使用PyTorch的AdamW优化器
        lr_scheduler_type="cosine",  # 使用余弦学习率调度
        weight_decay=0.01,          # 权重衰减
        max_grad_norm=1.0,          # 梯度裁剪
    )

    # 9. 开始训练
    logger.info("🎯 开始训练...")
    trainer_stats = unsloth_train(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        formatting_prompts_func=formatting_prompts_func,
        training_args=training_args
    )

    # 10. 保存训练统计
    logger.info("💾 保存训练统计...")
    logger.save_training_stats(trainer_stats, "training_stats.json")

    logger.info("✅ 训练完成!")
    logger.info(f"📊 训练统计: {trainer_stats}")

if __name__ == "__main__":
    main() 