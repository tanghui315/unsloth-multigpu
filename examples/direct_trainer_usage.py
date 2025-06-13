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

# Directly import our core components
from unsloth_multigpu.core import AggregationMethod, MultiGPUTrainer
from unsloth_multigpu.utils import ConfigManager, DeviceManager, MultiGPULogger


def create_dataloader(dataset, tokenizer, batch_size=8):
    """Create a data loader"""
    def tokenize_function(examples):
        # Simplified tokenization function
        inputs = tokenizer(
            examples['instruction'], 
            truncation=True, 
            padding=True, 
            max_length=512,
            return_tensors="pt"
        )
        # Add labels (in practice, you may need more complex processing)
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
    """Main function for direct usage of MultiGPUTrainer"""
    
    # 1. Initialize device and config managers
    print("🔧 Initializing configuration...")
    device_manager = DeviceManager()
    config_manager = ConfigManager()
    logger = MultiGPULogger(log_dir="./logs/direct_trainer")
    
    # Check available devices
    available_devices = device_manager.get_available_devices()
    num_gpus = min(len(available_devices), 4)  # Use up to 4 GPUs
    
    if num_gpus < 2:
        print("⚠️ Multi-GPU training requires at least 2 GPUs, demo will run in single GPU mode")
        num_gpus = 1  # Fallback to single GPU mode
    
    print(f"📊 Training with {num_gpus} GPU(s)")
    
    # 2. Load model
    print("📥 Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        "unsloth/llama-2-7b-bnb-4bit",
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=True
    )
    
    # 3. Prepare data
    print("📚 Preparing dataset...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train[:100]")  # Use a small dataset for demo
    
    # Create data loader
    dataloader = create_dataloader(dataset, tokenizer, batch_size=4)
    
    # 4. Configure optimizer
    optimizer_config = {
        'class': torch.optim.AdamW,
        'kwargs': {
            'lr': 2e-5,
            'weight_decay': 0.01,
            'betas': (0.9, 0.999)
        }
    }
    
    # 5. Create MultiGPUTrainer
    print("🚀 Initializing MultiGPUTrainer...")
    trainer = MultiGPUTrainer(
        model=model,
        num_gpus=num_gpus,
        optimizer_config=optimizer_config,
        aggregation_method=AggregationMethod.MEAN,
        enable_gradient_checkpointing=True,
        enable_mixed_precision=True
    )
    
    # 6. Setup trainer
    trainer.setup()
    
    # 7. Start training
    print("🎯 Starting training...")
    
    try:
        # Train for multiple epochs
        for epoch in range(2):  # Train for 2 epochs as a demo
            print(f"\n📖 Epoch {epoch + 1}/2")
            
            epoch_stats = trainer.train_epoch(
                dataloader=dataloader,
                max_steps=10  # Limit steps for demo
            )
            
            print(f"✅ Epoch {epoch + 1} finished:")
            print(f"  - Average loss: {epoch_stats['avg_loss']:.4f}")
            print(f"  - Training steps: {epoch_stats['steps']}")
            print(f"  - Samples processed: {epoch_stats['samples']}")
            print(f"  - Epoch time: {epoch_stats['epoch_time']:.2f}s")
        
        # 8. Get training statistics
        print("\n📊 Training statistics:")
        training_stats = trainer.get_training_stats()
        
        print(f"  - Total steps: {training_stats['global_step']}")
        print(f"  - Batches processed: {training_stats['num_batches_processed']}")
        print(f"  - Samples processed: {training_stats['num_samples_processed']}")
        print(f"  - Avg forward time: {training_stats.get('avg_forward_time_ms', 0):.2f}ms")
        print(f"  - Avg backward time: {training_stats.get('avg_backward_time_ms', 0):.2f}ms")
        print(f"  - Avg aggregation time: {training_stats.get('avg_aggregation_time_ms', 0):.2f}ms")
        
        # 9. Save checkpoint
        checkpoint_path = "./checkpoints/direct_trainer_checkpoint.pt"
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        trainer.save_checkpoint(checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")
        
    except Exception as e:
        print(f"❌ Error occurred during training: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 10. Cleanup resources
        print("🧹 Cleaning up resources...")
        trainer.cleanup()
        print("✅ Direct MultiGPUTrainer usage example complete!")

def compare_approaches():
    """Compare two usage approaches"""
    print("\n" + "="*60)
    print("📋 Comparison of two usage approaches")
    print("="*60)
    
    print("\n🔧 Approach 1: Hook mechanism (quick_start.py)")
    print("  Advantages:")
    print("    ✅ Zero-intrusion - no need to modify existing code")
    print("    ✅ Backward compatible - fully compatible with Unsloth API")
    print("    ✅ Easy to use - just call enable_multi_gpu()")
    print("  Disadvantages:")
    print("    ⚠️ Less transparent - users may not know the internal mechanism")
    print("    ⚠️ Harder to debug - hook replacement may be difficult to debug")
    
    print("\n🎯 Approach 2: Direct usage (this example)")
    print("  Advantages:")
    print("    ✅ Full control - users know exactly what is being used")
    print("    ✅ Easier to debug - direct access to all components")
    print("    ✅ Flexible configuration - fine-tune all parameters")
    print("  Disadvantages:")
    print("    ⚠️ Learning required - users need to learn the new API")
    print("    ⚠️ Code modification - need to change existing training code")
    
    print("\n💡 Recommendation:")
    print("  - For quick migration: use the Hook mechanism")
    print("  - For new projects/advanced users: use MultiGPUTrainer directly")
    print("  - For debugging/customization: use MultiGPUTrainer directly")

if __name__ == "__main__":
    main()
    compare_approaches() 