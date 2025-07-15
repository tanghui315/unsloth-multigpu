"""
SeleKT Algorithm Training Example
Demonstrates how to use SeleKT with Unsloth Multi-GPU for sparse LoRA parameter training
"""

import torch
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

import unsloth_multigpu as ump


def main():
    """Main training function with SeleKT integration"""
    
    print("🚀 SeleKT + Unsloth Multi-GPU Training Example")
    
    # 1. Model configuration
    model_name = "microsoft/DialoGPT-small"  # Model to train
    max_seq_length = 2048
    
    # 💡 Note: SeleKT now focuses on LoRA parameters
    # - LoRA parameters start from 0 and represent learned deltas
    # - SeleKT applies sparse selection directly to these parameters
    # - No need for base model comparison
    
    # 2. Enable SeleKT algorithm (before enabling multi-GPU)
    print("🧠 Enabling SeleKT algorithm...")
    ump.enable_selekt(
        alpha=0.05,  # Keep only top 5% of LoRA parameter changes
        apply_on_save=True,  # Apply SeleKT when saving checkpoints
        save_selekt_checkpoints=True,  # Save SeleKT-processed models
        apply_frequency=1  # Apply every save
    )
    
    # 3. Enable multi-GPU support
    print("🔧 Enabling multi-GPU support...")
    ump.enable_multi_gpu(
        num_gpus=2,
        batch_size_per_gpu=2,
        ddp_backend="nccl",
        enable_memory_optimization=True
    )
    
    # 4. Load model and tokenizer
    print("📥 Loading model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=True,  # Use 4-bit quantization to save memory
    )
    
    # 5. Configure LoRA
    print("⚙️ Configuring LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    # 6. Prepare dataset (use your own dataset here)
    print("📊 Preparing dataset...")
    # This is a placeholder - replace with your actual dataset loading
    dataset = [
        {"text": f"Example training text {i}. " * 50} 
        for i in range(1000)  # Small dataset for demo
    ]
    
    class SimpleDataset:
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]
    
    train_dataset = SimpleDataset(dataset)
    
    # 7. Configure training arguments
    training_args = TrainingArguments(
        output_dir="./selekt_results",
        num_train_epochs=2,
        per_device_train_batch_size=2,
        learning_rate=2e-5,
        warmup_steps=10,
        logging_steps=5,
        save_strategy="steps",
        save_steps=50,  # Save frequently to see SeleKT in action
        save_total_limit=5,
        # DDP optimizations
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )
    
    # 8. Create trainer (SeleKT will be automatically added via hooks)
    print("🔧 Creating trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        args=training_args,
        max_seq_length=max_seq_length,
    )
    
    # 9. Check status before training
    print("\n📋 System Status:")
    multi_gpu_status = ump.get_multi_gpu_status()
    selekt_status = ump.get_selekt_status()
    
    print("Multi-GPU:", {
        'enabled': multi_gpu_status['enabled'],
        'num_gpus': multi_gpu_status['num_gpus'],
        'gpu_info': multi_gpu_status.get('gpu_info', {})
    })
    print("SeleKT:", {
        'enabled': selekt_status['selekt_enabled'],
        'config': selekt_status['config']
    })
    
    # 10. Start training (SeleKT will be applied automatically on saves)
    print("\n🚀 Starting training with SeleKT...")
    print("SeleKT will automatically:")
    print("  • 🎯 Process only LoRA parameters (efficient)")
    print("  • 📊 Apply sparse parameter selection at each save")
    print("  • 💾 Keep only top 5% of LoRA parameter values") 
    print("  • 💿 Save both regular and SeleKT checkpoints")
    print("  • 🚀 No base model loading required (memory efficient)")
    
    try:
        trainer_stats = trainer.train()
        print(f"✅ Training completed: {trainer_stats}")
        
        # 显示内存使用情况
        if torch.cuda.is_available():
            print(f"💾 GPU Memory: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise
    
    finally:
        # 11. Cleanup
        print("🧹 Cleaning up...")
        ump.disable_selekt()
        ump.disable_multi_gpu()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("✅ Training example completed!")

def standalone_selekt_example():
    """Example of using SeleKT as a standalone function"""
    print("\n🔬 Standalone SeleKT Example")
    
    # Load your trained model with LoRA
    model, _ = FastLanguageModel.from_pretrained(
        "microsoft/DialoGPT-small",
        load_in_4bit=True
    )
    
    # Add LoRA to the model
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    
    # Apply SeleKT directly to LoRA parameters
    from unsloth_multigpu.algorithms import apply_selekt_to_model
    
    print("🔧 Applying SeleKT to LoRA parameters...")
    selekt_model = apply_selekt_to_model(
        model=model,
        alpha=0.05  # Keep top 5% of LoRA parameters
    )
    
    # Save the SeleKT-processed model
    selekt_model.save_pretrained("./selekt_processed_model")
    print("✅ SeleKT applied to LoRA parameters and saved!")

def advanced_selekt_config():
    """Advanced SeleKT configuration example"""
    print("\n⚙️ Advanced SeleKT Configuration")
    
    # Custom SeleKT configuration
    from unsloth_multigpu.algorithms import create_selekt_config

    # 配置1：高稀疏度模式
    high_sparsity_config = create_selekt_config(
        alpha=0.02,  # Keep only top 2% (higher sparsity)
        apply_on_save=True,
        save_selekt_checkpoints=True,
        apply_frequency=1,  # Apply every save
        memory_efficient=True
    )
    
    print(f"High sparsity config: {high_sparsity_config.__dict__}")
    
    # 配置2：低稀疏度模式
    low_sparsity_config = create_selekt_config(
        alpha=0.1,  # Keep top 10% (lower sparsity)
        apply_on_save=True,
        save_selekt_checkpoints=True,
        apply_frequency=2,  # Apply every 2 saves
        memory_efficient=True
    )
    
    print(f"Low sparsity config: {low_sparsity_config.__dict__}")
    
    # 配置3：仅在训练结束时应用
    final_only_config = create_selekt_config(
        alpha=0.05,
        apply_on_save=False,  # 不在每次保存时应用
        save_selekt_checkpoints=True,
        memory_efficient=True
    )
    
    print(f"Final only config: {final_only_config.__dict__}")

def memory_usage_demo():
    """演示内存使用情况"""
    print("\n💾 Memory Usage Demo")
    
    import gc

    import torch

    # 清理内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    print(f"Initial GPU memory: {initial_memory/1024**3:.2f} GB")
    
    # 加载量化模型
    model, tokenizer = FastLanguageModel.from_pretrained(
        "microsoft/DialoGPT-small",
        load_in_4bit=True
    )
    
    model_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    print(f"After loading quantized model: {model_memory/1024**3:.2f} GB")
    
    # 添加LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    
    lora_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    print(f"After adding LoRA: {lora_memory/1024**3:.2f} GB")
    print(f"LoRA overhead: {(lora_memory-model_memory)/1024**3:.2f} GB")
    
    # 启用SeleKT（无额外内存开销）
    ump.enable_selekt(
        alpha=0.05,
        apply_on_save=True,
        save_selekt_checkpoints=True
    )
    
    selekt_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    print(f"After enabling SeleKT: {selekt_memory/1024**3:.2f} GB")
    print(f"SeleKT overhead: {(selekt_memory-lora_memory)/1024**3:.2f} GB")
    
    # 清理
    ump.disable_selekt()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    
    print("Choose example to run:")
    print("1. Full training with SeleKT + Multi-GPU")
    print("2. Standalone SeleKT application") 
    print("3. Advanced configuration")
    print("4. Memory usage demo")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        main()
    elif choice == "2":
        standalone_selekt_example()
    elif choice == "3":
        advanced_selekt_config()
    elif choice == "4":
        memory_usage_demo()
    else:
        print("Invalid choice. Running main example...")
        main()