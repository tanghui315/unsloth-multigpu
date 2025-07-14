import os
import sys
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional

import torch
from datasets import load_dataset
# Import dataset loading function
from llamafactory.data import get_dataset
from llamafactory.hparams import DataArguments, ModelArguments
from transformers import Seq2SeqTrainingArguments, TrainingArguments
from trl import SFTConfig, SFTTrainer
# Import unsloth related modules first to avoid warnings
from unsloth import FastLanguageModel

import unsloth_multigpu as unsloth_multigpu

batch_size_per_gpu = 2
# Define necessary argument classes
@dataclass
class ModelArguments(ModelArguments):
    model_name_or_path: str = "Qwen/Qwen2.5-Coder-32B-Instruct"
    trust_remote_code: bool = True
    use_fast_tokenizer: bool = True
    cache_dir: str = None 
    hf_hub_token: str = None

@dataclass
class DataArguments(DataArguments):
    dataset_dir: str = "/data/V2_2_530/"
    dataset: List[str] = field(default_factory=lambda: ["dataset_name"])
    split: str = "train"
    streaming: bool = False
    val_size: float = 0.05
    buffer_size: int = 16384
    cutoff_len: int = 16384
    overwrite_cache: bool = True
    preprocessing_num_workers: int = 8
    max_seq_length: int = 16384
    template: str = "qwen"
    tokenized_path: Optional[str] = None
    train_on_prompt: bool = False

def main():
    # 1. Enable multi-GPU support
    print("🚀 Enable multi-GPU support...")
    unsloth_multigpu.enable_multi_gpu(
        num_gpus=2,  # Use 4 GPUs
        debug=True,
        batch_size_per_gpu=batch_size_per_gpu,  # Batch size per GPU
        ddp_backend="nccl"  # Use NCCL for GPU communication
    )
    
    model_args = ModelArguments()
    data_args = DataArguments()
    training_args =  SFTConfig(
            dataset_text_field=None, 
            output_dir="./outputs",
            per_device_train_batch_size = batch_size_per_gpu,
            save_strategy = "steps",   # Save by steps
            save_steps     = 10000,      # Save every 10000 global steps
            gradient_accumulation_steps = 1, # Use GA to mimic batch size!
            warmup_ratio = 0.1,
            num_train_epochs = 1, # Set this for 1 full training run.
            learning_rate = 2e-5, # Normal learning rate
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            report_to = "none", # Use this for WandB etc
            dataloader_num_workers=2,
            dataloader_pin_memory=True,
            remove_unused_columns=False,
        )

    # 👉 Additionally create a HF Seq2SeqTrainingArguments for LlamaFactory data processing
    hf_training_args = Seq2SeqTrainingArguments(
            output_dir="./outputs",
            per_device_train_batch_size=batch_size_per_gpu,
            predict_with_generate=False,
            do_train=True,
            do_eval=True,
            logging_steps=1,
            learning_rate=2e-5,  # Keep consistent with training_args
            num_train_epochs=1,
            seed=3407,
            overwrite_output_dir=True,
        )

    # 🔍 Debug info - show initialized arguments
    print("🔍 Debug ModelArguments:")
    from pprint import pprint
    pprint(model_args)
    print("🔍 Debug DataArguments:")
    pprint(data_args)
    # 2. Check system status
    status = unsloth_multigpu.get_multi_gpu_status()
    print(f"📊 Multi-GPU status: {status}")

    from llamafactory.data import get_template_and_fix_tokenizer

    # 3. Load model
    print("📥 Load model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_args.model_name_or_path,  # Use 4bit quantized version
        max_seq_length=data_args.max_seq_length,
        dtype=torch.bfloat16,  # Use torch dtype object instead of string
        load_in_4bit=True
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,           # Choose any number > 0! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 32,  # Best to choose alpha = rank or rank*2
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,   # We support rank stabilized LoRA
        loftq_config = None,  # And LoftQ
    )

    # 4. Prepare dataset
    print("📚 Prepare dataset...")
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    # Use hf_training_args instead of SFTConfig to meet LlamaFactory preprocessing requirements
    dataset_module = get_dataset(template, model_args, data_args, hf_training_args, "sft", tokenizer, None)

    # 🔍 Debug info - show dataset_module content
    print(f"🔍 Debug dataset_module:")
    print(f"   Type: {type(dataset_module)}")
    if isinstance(dataset_module, dict):
        print(f"   Keys: {list(dataset_module.keys())}")
        for key, value in dataset_module.items():
            print(f"   {key}: {type(value)} - Length: {len(value) if hasattr(value, '__len__') else 'N/A'}")
    else:
        print(f"   Content: {dataset_module}")
        if hasattr(dataset_module, '__dict__'):
            print(f"   Attributes: {list(dataset_module.__dict__.keys())}")

    # 4.1 Create custom Data Collator (remove labels first, then pad)
    def sft_data_collator(features, tokenizer, max_len):
         """Pad feature list, derive labels from input_ids, set padding part to -100 to ignore loss"""
         IGNORE_INDEX = -100
         # 1) Extract labels and construct input with only tokenizer supported fields
         model_keys = tokenizer.model_input_names  # Usually includes input_ids/attention_mask/token_type_ids
         labels_list = []
         input_features = []
         for feat in features:
             # Save labels
             labels_list.append(feat["labels"])

             # Filter out None or fields not recognized by tokenizer to avoid convert_to_tensors error
             filtered = {k: v for k, v in feat.items() if k in model_keys and v is not None}
             input_features.append(filtered)

         # 2) Use tokenizer to pad
         batch = tokenizer.pad(
             input_features,
             padding=True,
             max_length=max_len,
             return_tensors="pt",
         )

         # 3) Process labels: first create a matrix full of -100, then overwrite valid part
         seq_len = batch["input_ids"].shape[1]
         new_labels = torch.full((len(labels_list), seq_len), IGNORE_INDEX, dtype=torch.long)
         for i, lbl in enumerate(labels_list):
             lbl = lbl[:seq_len]
             new_labels[i, :len(lbl)] = torch.tensor(lbl, dtype=torch.long)
         batch["labels"] = new_labels

         return batch

    data_collator = partial(sft_data_collator, tokenizer=tokenizer, max_len=data_args.max_seq_length)

    # 🐞 Debug check if labels padding is -100
    debug_samples = [dataset_module["train_dataset"][i] for i in range(2)]
    debug_batch = data_collator(debug_samples)
    print("🔍 labels tail debug:", debug_batch["labels"][0][-20:])

    # 5. Configure training parameters
    print("⚙️ Configure training parameters...")


    # 6. Start training
    print("🎯 Start training...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset_module["train_dataset"],
        eval_dataset = dataset_module["eval_dataset"],  # Optional evaluation
        args = training_args,
        data_collator = data_collator,
    )
    trainer_stats = trainer.train()
    print("✅ Training completed!")
    print(f"📊 Training statistics: {trainer_stats}")

if __name__ == "__main__":
    main() 