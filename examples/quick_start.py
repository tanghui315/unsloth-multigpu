import os

import torch
# 确保首先导入 unsloth 以避免性能警告
import unsloth
from datasets import load_dataset
from transformers import TrainingArguments
from unsloth import FastLanguageModel, unsloth_train

import unsloth_multigpu as unsloth_multigpu


def main():
    batch_size_per_gpu = 2
    # 1. Enable multi-GPU support
    print("🚀 Enable multi-GPU support...")
    unsloth_multigpu.enable_multi_gpu(
        num_gpus=2,  # Use 4 GPUs
        batch_size_per_gpu=batch_size_per_gpu,  # Batch size per GPU
        gradient_aggregation="mean"  # Gradient aggregation strategy
    )

    # 2. Check system status
    status = unsloth_multigpu.get_multi_gpu_status()
    print(f"📊 Multi-GPU status: {status}")

    # 3. Load model
    print("📥 Load model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        "/home/valiantsec/cjr/models/Qwen/Qwen2.5-Coder-7B-Instruct/",  # Use 4bit quantized version
        max_seq_length=4096,
        dtype=torch.bfloat16,
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
        per_device_train_batch_size=batch_size_per_gpu,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        save_strategy = "steps",
        save_steps=100,
        fp16=True,
        logging_steps=1,
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