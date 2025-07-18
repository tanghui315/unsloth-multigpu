#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRPO Multi-GPU Training Example for unsloth-multigpu

This example demonstrates how to use GRPO (Generalized Reinforcement Learning from Policy Optimization)
with multi-GPU training support. Based on the original Qwen2.5 3B GRPO notebook from Unsloth.

Usage:
    # Single GPU
    python examples/grpo_multi_gpu_training.py
    
    # Multi-GPU with torchrun
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 examples/grpo_multi_gpu_training.py

Features:
    - Multi-GPU GRPO training with DDP
    - Distributed reward computation
    - vLLM fast inference support
    - Multiple reward functions
    - Memory optimization
"""

import os
import re
import torch
from typing import List
from datasets import load_dataset, Dataset

# Import unsloth-multigpu
import unsloth_multigpu as ump

# Import Unsloth
from unsloth import FastLanguageModel, is_bfloat16_supported

def setup_model_and_tokenizer():
    """Setup model and tokenizer with memory optimization"""
    max_seq_length = 1024  # Can increase for longer reasoning traces
    lora_rank = 64  # Larger rank = smarter, but slower
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        max_seq_length=max_seq_length,
        load_in_4bit=True,  # 4-bit quantization for memory efficiency
        fast_inference=True,  # Enable vLLM fast inference
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.5,  # Reduce if out of memory
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_rank,
        use_gradient_checkpointing="unsloth",  # Enable long context finetuning
        random_state=3407,
    )
    
    return model, tokenizer

def prepare_dataset():
    """Prepare GSM8K dataset for GRPO training"""
    
    SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""
    
    def extract_hash_answer(text: str) -> str:
        """Extract answer from GSM8K format"""
        if "####" not in text:
            return ""
        return text.split("####")[1].strip()
    
    def get_gsm8k_questions(split="train") -> Dataset:
        """Load and format GSM8K dataset"""
        data = load_dataset('openai/gsm8k', 'main')[split]
        data = data.map(lambda x: {
            'prompt': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': x['question']}
            ],
            'answer': extract_hash_answer(x['answer'])
        })
        return data
    
    dataset = get_gsm8k_questions()
    print(f"📊 Dataset loaded: {len(dataset)} samples")
    
    return dataset

def define_reward_functions():
    """Define reward functions for GRPO training"""
    
    def extract_xml_answer(text: str) -> str:
        """Extract answer from XML format"""
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()
    
    def correctness_reward_func(prompts, completions, answer, **kwargs) -> List[float]:
        """Reward function for answer correctness"""
        responses = [completion[0]['content'] for completion in completions]
        q = prompts[0][-1]['content']
        extracted_responses = [extract_xml_answer(r) for r in responses]
        
        # Rank 0 only logging to avoid duplicates
        if not hasattr(torch.distributed, 'is_initialized') or not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
        
        return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]
    
    def int_reward_func(completions, **kwargs) -> List[float]:
        """Reward function for integer answers"""
        responses = [completion[0]['content'] for completion in completions]
        extracted_responses = [extract_xml_answer(r) for r in responses]
        return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]
    
    def strict_format_reward_func(completions, **kwargs) -> List[float]:
        """Reward function for strict format compliance"""
        pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, r) for r in responses]
        return [0.5 if match else 0.0 for match in matches]
    
    def soft_format_reward_func(completions, **kwargs) -> List[float]:
        """Reward function for soft format compliance"""
        pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, r, re.DOTALL) for r in responses]
        return [0.5 if match else 0.0 for match in matches]
    
    def count_xml(text) -> float:
        """Count XML tags and penalize extra content"""
        count = 0.0
        if text.count("<reasoning>\n") == 1:
            count += 0.125
        if text.count("\n</reasoning>\n") == 1:
            count += 0.125
        if text.count("\n<answer>\n") == 1:
            count += 0.125
            count -= len(text.split("\n</answer>\n")[-1])*0.001
        if text.count("\n</answer>") == 1:
            count += 0.125
            count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
        return count
    
    def xmlcount_reward_func(completions, **kwargs) -> List[float]:
        """Reward function for XML tag counting"""
        contents = [completion[0]["content"] for completion in completions]
        return [count_xml(c) for c in contents]
    
    return [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ]

def create_grpo_trainer(model, tokenizer, dataset, reward_funcs):
    """Create GRPO trainer with optimized configuration"""
    from trl import GRPOConfig, GRPOTrainer
    
    # Check if in DDP environment and adjust batch size accordingly
    world_size = 1
    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])
    
    # Adjust batch size and generation settings for multi-GPU
    per_device_batch_size = 1
    num_generations = 8
    
    if world_size > 1:
        # In DDP mode, reduce generations per GPU to avoid memory issues
        num_generations = max(4, num_generations // world_size)
        print(f"🔧 DDP detected: {world_size} processes, using {num_generations} generations per GPU")
    
    training_args = GRPOConfig(
        use_vllm=True,  # use vLLM for fast inference
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=1,
        num_generations=num_generations,
        max_prompt_length=256,
        max_completion_length=200,
        max_steps=50,  # Reduced for demo purposes
        save_steps=50,
        max_grad_norm=0.1,
        report_to="none",
        output_dir="outputs/grpo_multi_gpu",
    )
    
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
    )
    
    return trainer

def main():
    """Main training function"""
    print("🚀 Starting GRPO Multi-GPU Training Example")
    print(f"📍 Process info: PID={os.getpid()}")
    
    # Check if in DDP environment
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        print(f"🌐 DDP Environment: Rank {rank}/{world_size}, Local Rank {local_rank}")
    else:
        print("🔄 Single GPU Environment")
    
    # Enable unsloth-multigpu support
    print("\n📦 Enabling unsloth-multigpu support...")
    
    # Enable GRPO support first
    grpo_success = ump.enable_grpo_support()
    if grpo_success:
        print("✅ GRPO hooks enabled successfully")
    else:
        print("❌ GRPO hooks failed to enable")
        return
    
    # Enable multi-GPU support
    try:
        ump.enable_multi_gpu(
            num_gpus=2,  # Will be ignored in DDP mode
            batch_size_per_gpu=1,
            ddp_backend="nccl",
            enable_memory_optimization=True,
            enable_gradient_checkpointing=True
        )
        print("✅ Multi-GPU support enabled")
    except Exception as e:
        print(f"⚠️ Multi-GPU setup: {e}")
        print("🔄 Continuing in single GPU mode")
    
    # Setup model and data
    print("\n🔧 Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer()
    print("✅ Model and tokenizer ready")
    
    print("\n📊 Preparing dataset...")
    dataset = prepare_dataset()
    print("✅ Dataset ready")
    
    print("\n🎯 Defining reward functions...")
    reward_funcs = define_reward_functions()
    print(f"✅ {len(reward_funcs)} reward functions defined")
    
    print("\n🏋️ Creating GRPO trainer...")
    trainer = create_grpo_trainer(model, tokenizer, dataset, reward_funcs)
    print("✅ GRPO trainer ready")
    
    # Display system status
    print("\n📋 System Status:")
    status = ump.get_multi_gpu_status()
    print(f"   Multi-GPU enabled: {status['enabled']}")
    if status['enabled']:
        print(f"   Number of GPUs: {status['num_gpus']}")
        print(f"   Configuration: {status['active_config']}")
    
    grpo_status = ump.get_grpo_status()
    print(f"   GRPO support: {grpo_status.get('grpo_hooks_applied', False)}")
    
    # Start training
    print("\n🚀 Starting GRPO training...")
    print("💡 The goal is to see the 'reward' column increase over steps!")
    print("⏳ You might need to wait 20-50 steps for meaningful rewards.")
    print("🔍 Watch for format compliance and correctness improvements.\n")
    
    try:
        trainer.train()
        print("✅ GRPO training completed successfully!")
        
        # Save the trained LoRA
        print("\n💾 Saving trained LoRA...")
        model.save_lora("outputs/grpo_multi_gpu_lora")
        print("✅ LoRA saved to outputs/grpo_multi_gpu_lora")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise
    finally:
        # Clean up
        print("\n🧹 Cleaning up...")
        ump.disable_grpo_support()
        ump.disable_multi_gpu()
        print("✅ Cleanup completed")

def test_inference():
    """Test inference with the trained model"""
    print("\n🧪 Testing inference...")
    
    # Load model for inference
    model, tokenizer = setup_model_and_tokenizer()
    
    # Test without LoRA
    text = tokenizer.apply_chat_template([
        {"role": "user", "content": "How many r's are in strawberry?"},
    ], tokenize=False, add_generation_prompt=True)
    
    try:
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=512,
        )
        
        print("🔍 Testing without GRPO training...")
        output = model.fast_generate(
            [text],
            sampling_params=sampling_params,
            lora_request=None,
        )[0].outputs[0].text
        print(f"Output without training:\n{output}\n")
        
        # Test with trained LoRA if it exists
        try:
            print("🔍 Testing with GRPO-trained LoRA...")
            lora_path = "outputs/grpo_multi_gpu_lora"
            if os.path.exists(lora_path):
                output_with_lora = model.fast_generate(
                    [text],
                    sampling_params=sampling_params,
                    lora_request=model.load_lora(lora_path),
                )[0].outputs[0].text
                print(f"Output with GRPO training:\n{output_with_lora}\n")
            else:
                print("⚠️ No trained LoRA found, skipping comparison")
                
        except Exception as e:
            print(f"⚠️ LoRA inference test failed: {e}")
            
    except ImportError:
        print("⚠️ vLLM not available, skipping inference test")
    except Exception as e:
        print(f"❌ Inference test failed: {e}")

if __name__ == "__main__":
    # Check for test mode
    if len(os.sys.argv) > 1 and os.sys.argv[1] == "--test-inference":
        test_inference()
    else:
        main()