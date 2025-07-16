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

    # Config 1: High sparsity + step trigger
    high_sparsity_config = create_selekt_config(
        alpha=0.02,  # Keep only top 2% (higher sparsity)
        apply_on_save=True,
        save_selekt_checkpoints=True,
        apply_frequency=1,  # Apply every save
        step_frequency=1000,  # Also apply every 1000 steps
        memory_efficient=True
    )
    
    print(f"High sparsity + step trigger config: {high_sparsity_config.__dict__}")
    
    # Config 2: Solution for large save_steps
    large_save_steps_config = create_selekt_config(
        alpha=0.05,
        apply_on_save=True,
        save_selekt_checkpoints=True,
        apply_frequency=1,  # Apply every save
        step_frequency=2000,  # Apply every 2000 steps independently
        max_interval_steps=5000,  # Never exceed 5000 steps without applying
        memory_efficient=True
    )
    
    print(f"Large save_steps solution: {large_save_steps_config.__dict__}")
    
    # Config 3: Step-only trigger mode
    step_only_config = create_selekt_config(
        alpha=0.05,
        apply_on_save=False,  # Do not apply on save
        save_selekt_checkpoints=False,
        step_frequency=1500,  # Apply every 1500 steps
        memory_efficient=True
    )
    
    print(f"Step-only trigger config: {step_only_config.__dict__}")
    
    # Config 4: Max interval protection
    max_interval_config = create_selekt_config(
        alpha=0.05,
        apply_on_save=True,
        save_selekt_checkpoints=True,
        apply_frequency=5,  # Apply every 5 saves
        max_interval_steps=3000,  # But must apply at least every 3000 steps
        memory_efficient=True
    )
    
    print(f"Max interval protection: {max_interval_config.__dict__}")
    
    # Config 5: Epoch-based trigger (recommended)
    epoch_based_config = create_selekt_config(
        alpha=0.05,
        epoch_frequency=1,           # Apply at the end of every epoch
        apply_on_save=False,         # Do not apply again on save
        save_selekt_checkpoints=True,
        memory_efficient=True
    )
    
    print(f"Epoch-based trigger config: {epoch_based_config.__dict__}")
    
    # Config 6: Hybrid Epoch + Safety interval
    hybrid_epoch_config = create_selekt_config(
        alpha=0.05,
        epoch_frequency=1,           # Apply at the end of every epoch
        max_interval_steps=2000,     # Safety net: must apply at least every 2000 steps
        apply_on_save=True,          # Also apply on save
        save_selekt_checkpoints=True,
        memory_efficient=True
    )
    print(f"Hybrid epoch + safety config: {hybrid_epoch_config.__dict__}")

def solve_large_save_steps_problem():
    """Demonstrate how to solve the large save_steps problem"""
    print("\n🎯 Solving the large save_steps problem")
    
    # Problem scenario
    print("❌ Problem scenario:")
    print("   save_steps = 10000")
    print("   apply_frequency = 1")
    print("   Result: SeleKT is only applied every 10000 steps")
    print()
    
    # Solution 1: Add step trigger
    print("✅ Solution 1 - Add step trigger:")
    print("   ump.enable_selekt(")
    print("       alpha=0.05,")
    print("       apply_frequency=1,      # Apply on every save")
    print("       step_frequency=2000,    # Also apply every 2000 steps")
    print("   )")
    print("   Effect: SeleKT is applied every 2000 steps, regardless of save_steps")
    print()
    
    # Solution 2: Max interval protection
    print("✅ Solution 2 - Max interval protection:")
    print("   ump.enable_selekt(")
    print("       alpha=0.05,")
    print("       apply_frequency=3,        # Apply every 3 saves")
    print("       max_interval_steps=5000,  # Must apply at least every 5000 steps")
    print("   )")
    print("   Effect: Even if save_steps is large, SeleKT will not go more than 5000 steps without being applied")
    print()
    
    # Solution 3: Hybrid mode
    print("✅ Solution 3 - Hybrid mode:")
    print("   ump.enable_selekt(")
    print("       alpha=0.05,")
    print("       apply_frequency=1,        # Apply on every save")
    print("       step_frequency=1500,      # Also apply every 1500 steps")
    print("       max_interval_steps=3000,  # Must apply at least every 3000 steps")
    print("   )")
    print("   Effect: Triple protection to ensure SeleKT is applied in a timely manner")

def demonstrate_trigger_scenarios():
    """Demonstrate different trigger scenarios"""
    print("\n📊 Trigger scenario demonstration")
    
    scenarios = [
        {
            "name": "Standard mode",
            "save_steps": 1000,
            "apply_frequency": 1,
            "step_frequency": None,
            "max_interval": None
        },
        {
            "name": "Large save_steps + step trigger",
            "save_steps": 10000,
            "apply_frequency": 1,
            "step_frequency": 2000,
            "max_interval": None
        },
        {
            "name": "Max interval protection",
            "save_steps": 8000,
            "apply_frequency": 3,
            "step_frequency": None,
            "max_interval": 5000
        },
        {
            "name": "Hybrid mode",
            "save_steps": 5000,
            "apply_frequency": 2,
            "step_frequency": 1500,
            "max_interval": 3000
        }
    ]
    
    for scenario in scenarios:
        print(f"\n🎯 {scenario['name']}:")
        print(f"   Config: save_steps={scenario['save_steps']}, apply_freq={scenario['apply_frequency']}")
        if scenario['step_frequency']:
            print(f"        step_frequency={scenario['step_frequency']}")
        if scenario['max_interval']:
            print(f"        max_interval={scenario['max_interval']}")
        
        # Simulate trigger timeline for the first 15000 steps
        print("   Trigger timeline:")
        simulate_triggers(
            save_steps=scenario['save_steps'],
            apply_frequency=scenario['apply_frequency'],
            step_frequency=scenario['step_frequency'],
            max_interval=scenario['max_interval'],
            total_steps=15000
        )

def simulate_triggers(save_steps, apply_frequency, step_frequency, max_interval, total_steps):
    """Simulate trigger timeline"""
    save_count = 0
    last_applied = 0
    triggers = []
    
    for step in range(1, total_steps + 1):
        # Check save trigger
        if step % save_steps == 0:
            save_count += 1
            should_apply_by_save = save_count % apply_frequency == 0
        else:
            should_apply_by_save = False
        
        # Check step trigger
        should_apply_by_step = False
        if step_frequency:
            should_apply_by_step = (step - last_applied) >= step_frequency
        
        # Check max interval trigger
        should_apply_by_max = False
        if max_interval:
            should_apply_by_max = (step - last_applied) >= max_interval
        
        # Decide whether to trigger
        if should_apply_by_save or should_apply_by_step or should_apply_by_max:
            reasons = []
            if should_apply_by_save:
                reasons.append("save")
            if should_apply_by_step:
                reasons.append("step")
            if should_apply_by_max:
                reasons.append("max_interval")
            
            triggers.append((step, reasons))
            last_applied = step
    
    # Show the first few triggers
    for i, (step, reasons) in enumerate(triggers[:5]):
        print(f"     Step {step:5d}: ✅ Apply SeleKT ({', '.join(reasons)})")
    
    if len(triggers) > 5:
        print(f"     ... (Total {len(triggers)} triggers)")
    
    print(f"     Average interval: {total_steps / len(triggers):.0f} steps" if triggers else "     No trigger")

def memory_usage_demo():
    """Demonstrate memory usage"""
    print("\n💾 Memory Usage Demo")
    
    import gc

    import torch

    # Clean up memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    print(f"Initial GPU memory: {initial_memory/1024**3:.2f} GB")
    
    # Load quantized model
    model, tokenizer = FastLanguageModel.from_pretrained(
        "microsoft/DialoGPT-small",
        load_in_4bit=True
    )
    
    model_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    print(f"After loading quantized model: {model_memory/1024**3:.2f} GB")
    
    # Add LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    
    lora_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    print(f"After adding LoRA: {lora_memory/1024**3:.2f} GB")
    print(f"LoRA overhead: {(lora_memory-model_memory)/1024**3:.2f} GB")
    
    # Enable SeleKT (no extra memory overhead)
    ump.enable_selekt(
        alpha=0.05,
        apply_on_save=True,
        save_selekt_checkpoints=True
    )
    
    selekt_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    print(f"After enabling SeleKT: {selekt_memory/1024**3:.2f} GB")
    print(f"SeleKT overhead: {(selekt_memory-lora_memory)/1024**3:.2f} GB")
    
    # Cleanup
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
    print("5. Solve large save_steps problem")
    print("6. Demonstrate trigger scenarios")
    
    choice = input("Enter choice (1-6): ").strip()
    
    if choice == "1":
        main()
    elif choice == "2":
        standalone_selekt_example()
    elif choice == "3":
        advanced_selekt_config()
    elif choice == "4":
        memory_usage_demo()
    elif choice == "5":
        solve_large_save_steps_problem()
    elif choice == "6":
        demonstrate_trigger_scenarios()
    else:
        print("Invalid choice. Running main example...")
        main()