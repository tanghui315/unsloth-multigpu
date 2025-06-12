'''
Author: tanghui315 tanghui315@126.com
Date: 2025-06-06 11:47:37
LastEditors: tanghui315 tanghui315@126.com
LastEditTime: 2025-06-12 09:51:32
FilePath: /us-train/unsloth_multigpu_prototype/examples/quick_start.py
Description: This is the default setting, please set `customMade`, open koroFileHeader to view and set: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
"""
Unsloth Multi-GPU Quick Start Example
This example demonstrates how to use Unsloth Multi-GPU for model training
"""

import os

import torch
from datasets import load_dataset
from transformers import TrainingArguments

import unsloth_multigpu_prototype as unsloth_multigpu
from unsloth import FastLanguageModel, unsloth_train


def main():
    # 1. Enable multi-GPU support
    print("🚀 Enable multi-GPU support...")
    unsloth_multigpu.enable_multi_gpu(
        num_gpus=4,  # Use 4 GPUs
        batch_size_per_gpu=8,  # Batch size per GPU
        gradient_aggregation="mean"  # Gradient aggregation strategy
    )

    # 2. Check system status
    status = unsloth_multigpu.get_multi_gpu_status()
    print(f"📊 Multi-GPU status: {status}")

    # 3. Load model
    print("📥 Load model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        "unsloth/llama-2-7b-bnb-4bit",  # Use 4bit quantized version
        max_seq_length=4096,
        dtype="bfloat16",
        load_in_4bit=True
    )

    # 4. Prepare dataset
    print("📚 Prepare dataset...")
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

    # 5. Configure training parameters
    print("⚙️ Configure training parameters...")
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        warmup_ratio=0.1,
    )

    # 6. Start training
    print("🎯 Start training...")
    trainer_stats = unsloth_train(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        formatting_prompts_func=formatting_prompts_func,
        training_args=training_args
    )

    print("✅ Training completed!")
    print(f"📊 Training statistics: {trainer_stats}")

if __name__ == "__main__":
    main() 