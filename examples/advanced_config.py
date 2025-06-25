"""
Unsloth Multi-GPU Advanced Configuration Example
This example demonstrates how to use Unsloth Multi-GPU's advanced features and configuration options
"""

import os

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

import unsloth_multigpu as unsloth_multigpu
from unsloth_multigpu.utils import ConfigManager, DeviceManager, MultiGPULogger


def main():
    # 1. Initialize logging system
    logger = MultiGPULogger(
        log_dir="./logs",
        log_level="INFO",
        enable_tensorboard=True
    )
    logger.info("🚀 Starting Unsloth Multi-GPU Advanced Configuration Example")

    # 2. Device management
    device_manager = DeviceManager()
    devices = device_manager.get_available_devices()
    logger.info(f"📊 Available GPUs: {devices}")

    # 3. Configuration management
    config_manager = ConfigManager()
    optimal_config = config_manager.get_optimal_config(
        model_size="7B",
        available_memory="32GB",
        num_gpus=len(devices)
    )
    logger.info(f"⚙️ Recommended config: {optimal_config}")

    # 4. Enable multi-GPU (using optimal config)
    logger.info("🔄 Enabling multi-GPU support...")
    unsloth_multigpu.enable_multi_gpu(
        **optimal_config,
        gradient_aggregation="weighted_mean",  # Use weighted mean aggregation
        memory_efficient=True,  # Enable memory optimization
        enable_profiling=True   # Enable performance profiling
    )

    # 5. Check system status
    status = unsloth_multigpu.get_multi_gpu_status()
    logger.info(f"📊 Multi-GPU status: {status}")

    # 6. Load model
    logger.info("📥 Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        "unsloth/llama-2-7b-bnb-4bit",
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=True
    )

    # 6.5. Configure LoRA
    logger.info("⚙️ Configuring LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # 7. Prepare dataset
    logger.info("📚 Preparing dataset...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    
    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['instruction'])):
            text = f"### Instruction:\n{example['instruction'][i]}\n\n"
            if example['input'][i]:
                text += f"### Input:\n{example['input'][i]}\n\n"
            text += f"### Response:\n{example['output'][i]}"
            output_texts.append(text)
        return {"text": output_texts}

    # Apply formatting to dataset
    dataset = dataset.map(formatting_prompts_func, batched=True, remove_columns=dataset.column_names)

    # 8. Configure training arguments
    logger.info("⚙️ Configuring training arguments...")
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
        # Advanced training config
        gradient_checkpointing=True,  # Enable gradient checkpointing
        optim="adamw_torch",         # Use PyTorch AdamW optimizer
        lr_scheduler_type="cosine",  # Use cosine learning rate scheduler
        weight_decay=0.01,           # Weight decay
        max_grad_norm=1.0,           # Gradient clipping
    )

    # 9. Create SFTTrainer
    logger.info("⚙️ Creating SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        args=training_args,
        max_seq_length=2048,
        packing=False,
    )

    # 10. Start training
    logger.info("🎯 Starting training...")
    trainer_stats = trainer.train()

    # 11. Save training statistics
    logger.info("💾 Saving training statistics...")
    logger.save_training_stats(trainer_stats, "training_stats.json")

    logger.info("✅ Training complete!")
    logger.info(f"📊 Training statistics: {trainer_stats}")

if __name__ == "__main__":
    main() 