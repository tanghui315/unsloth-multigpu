#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRPO Multi-GPU Training Script - Pure YAML Configuration

This script reads ALL parameters from YAML configuration files only.
No command line arguments are needed or processed.
"""

import logging
import os
import sys
import types
import yaml
import torch
import transformers
from typing import Optional, List, Dict, Optional, Union
from dataclasses import dataclass, field
from datasets import load_dataset, Dataset, DatasetDict
from transformers import set_seed
from pathlib import Path
import json
# Initialize logger first
logging.basicConfig(level=logging.INFO)
_basic_logger = logging.getLogger(__name__)

# Import DDP logger after unsloth_multigpu import
try:
    from unsloth_multigpu.utils.ddp_logging import get_ddp_logger
    logger = get_ddp_logger(__name__)
    USE_DDP_LOGGER = True
except ImportError:
    # Fallback to basic logger if DDP logger not available
    logger = _basic_logger
    USE_DDP_LOGGER = False

# Import unsloth-multigpu
import unsloth_multigpu as ump

# Import Unsloth components
from unsloth import FastLanguageModel
from unsloth.models.llama import FastLlamaModel
try:
    from unsloth_zoo.utils import _get_dtype
except ImportError:
    def _get_dtype(torch_dtype):
        if torch_dtype == torch.float16:
            return torch.float16
        elif torch_dtype == torch.bfloat16:
            return torch.bfloat16
        else:
            return torch.float16

# Import TRL and open-r1
from trl import GRPOTrainer, GRPOConfig

# Try to import open-r1 reward functions directly
try:
    # Import professional reward functions directly from open_r1.rewards
    from open_r1.rewards import get_reward_funcs
    OPEN_R1_REWARDS_AVAILABLE = True
    logger.info("✅ open-r1 professional reward functions loaded successfully")
except ImportError as e:
    OPEN_R1_REWARDS_AVAILABLE = False
    logger.warning(f"⚠️ open-r1 rewards not available: {e}")
    
    

if OPEN_R1_REWARDS_AVAILABLE:
    logger.info("✅ Full open-r1 components loaded successfully")
else:
    logger.info("📦 Falling back to basic TRL implementation")


def make_conversation(example, prompt_column = "problem"):
    prompt = []

    prompt.append({"role": "system", "content": "Answer the question step by step, showing your reasoning. then provides the answer to the user. The reasoning process should be enclosed in <think> </think> tags,like this: <think>This is the reasoning process</think>The final answer is: <answer>the answer</answer>"})

    if prompt_column not in example:
        raise ValueError(f"Dataset Question Field Error: {prompt_column} is not supported.")

    prompt.append({"role": "user", "content": example[prompt_column]})
    return {"prompt": prompt}


def load_dataset_from_source(
    source: Union[str, Path],
    data_format: str = "jsonl",
    split_ratio: float = 0.05,
    seed: int = 42,
    name: Optional[str] = None,
    **kwargs
) -> Dict[str, Dataset]:
    if data_format == "jsonl":
        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(f"Not found file: {source}")
        data = []
        with open(source, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {line.strip()}\nError: {e}")
        
        # Create Dataset object
        dataset = Dataset.from_list(data)
        
        # Split the dataset
        splits = dataset.train_test_split(
            test_size=split_ratio,
            seed=seed,
            shuffle=True
        )
        
        # Return DatasetDict instead of a regular dict
        # Note: The split names are fixed as 'train' and 'test' here
        return DatasetDict({
            "train": splits["train"],
            "test": splits["test"]
        })
    
    elif data_format == "hf":
        # Load HuggingFace dataset
        # Note: HuggingFace datasets may have different split names, such as 'train', 'validation', 'test', etc.
        dataset = load_dataset(source, name=name, **kwargs)
        return dataset
    
    else:
        raise ValueError(f"Unsupported data format: {data_format}")


def load_yaml_config(config_file_path: str) -> dict:
    """Load and validate YAML configuration file"""
    if not os.path.exists(config_file_path):
        logger.error(f"❌ Configuration file not found: {config_file_path}")
        sys.exit(1)
    
    try:
        with open(config_file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"✅ YAML configuration loaded from: {config_file_path}")
        return config
    except Exception as e:
        logger.error(f"❌ Failed to load YAML configuration: {e}")
        sys.exit(1)


def create_grpo_config_from_yaml(yaml_config: dict) -> GRPOConfig:
    """Create GRPOConfig from YAML configuration"""
    
    # Create a mapping of YAML keys to GRPOConfig field names
    yaml_to_grpo_mapping = {
        # Core training parameters
        'optim': 'optim',
        'learning_rate': 'learning_rate', 
        'lr_scheduler_type': 'lr_scheduler_type',
        'lr_scheduler_kwargs': 'lr_scheduler_kwargs',
        'warmup_ratio': 'warmup_ratio',
        'weight_decay': 'weight_decay',
        
        # Training configuration
        'num_train_epochs': 'num_train_epochs',
        'max_steps': 'max_steps',
        'per_device_train_batch_size': 'per_device_train_batch_size',
        'per_device_eval_batch_size': 'per_device_eval_batch_size',
        'gradient_accumulation_steps': 'gradient_accumulation_steps',
        
        # Memory and performance
        'gradient_checkpointing': 'gradient_checkpointing',
        'gradient_checkpointing_kwargs': 'gradient_checkpointing_kwargs',
        'bf16': 'bf16',
        'fp16': 'fp16',
        
        # GRPO specific
        'num_generations': 'num_generations',
        'max_length': 'max_length',
        'max_prompt_length': 'max_prompt_length',
        'max_completion_length': 'max_completion_length',
        
        # Data processing
        'packing': 'packing',
        
        # vLLM configuration (TRL standard parameters)
        'use_vllm': 'use_vllm',
        'vllm_mode': 'vllm_mode',
        'vllm_server_host': 'vllm_server_host',
        'vllm_server_port': 'vllm_server_port',
        'vllm_server_base_url': 'vllm_server_base_url',
        'vllm_server_timeout': 'vllm_server_timeout',
        'vllm_gpu_memory_utilization': 'vllm_gpu_memory_utilization',
        'vllm_tensor_parallel_size': 'vllm_tensor_parallel_size',
        'vllm_guided_decoding_regex': 'vllm_guided_decoding_regex',
        
        # Logging and saving
        'logging_steps': 'logging_steps',
        'logging_strategy': 'logging_strategy',
        'save_steps': 'save_steps',
        'save_strategy': 'save_strategy',
        'save_total_limit': 'save_total_limit',
        
        # Evaluation
        'do_eval': 'do_eval',
        'eval_strategy': 'eval_strategy',
        'eval_steps': 'eval_steps',
        
        # Output and reporting
        'output_dir': 'output_dir',
        'overwrite_output_dir': 'overwrite_output_dir',
        'report_to': 'report_to',
        
        # Reproducibility
        'seed': 'seed',
        'data_seed': 'data_seed',
    }
    
    # Start with default GRPOConfig
    grpo_config = GRPOConfig()
    
    # Apply YAML values to GRPOConfig with special handling for problematic parameters
    applied_params = []
    
    for yaml_key, value in yaml_config.items():
        if yaml_key in yaml_to_grpo_mapping:
            grpo_field = yaml_to_grpo_mapping[yaml_key]
            if hasattr(grpo_config, grpo_field):
                # Special handling for report_to parameter
                if yaml_key == 'report_to':
                    if isinstance(value, list) and len(value) == 1 and value[0] == 'none':
                        # Convert "none" to empty list to disable all reporting
                        processed_value = []
                        logger.info("🔧 Fixed report_to: converted 'none' to empty list to disable reporting")
                    elif isinstance(value, list) and 'none' in value:
                        # Remove 'none' from the list
                        processed_value = [v for v in value if v != 'none']
                        logger.info(f"🔧 Fixed report_to: removed 'none', kept: {processed_value}")
                    else:
                        processed_value = value
                else:
                    processed_value = value
                
                setattr(grpo_config, grpo_field, processed_value)
                applied_params.append(f"{yaml_key} -> {grpo_field}")
    
    logger.info(f"✅ Applied {len(applied_params)} parameters to GRPOConfig")
    for param in applied_params[:10]:  # Show first 10
        logger.info(f"   {param}")
    if len(applied_params) > 10:
        logger.info(f"   ... and {len(applied_params) - 10} more")
    
    return grpo_config


@dataclass
class SimpleScriptArgs:
    """Simple script arguments from YAML - fully compatible with open-r1 reward functions"""
    dataset_name: str
    dataset_config: Optional[str] = "main"
    dataset_train_split: str = "train"
    dataset_test_split: str = "test"
    dataset_num_proc: int = 4
    reward_funcs: List[str] = field(default_factory=lambda: ["accuracy", "format", "reasoning_steps"])
    
    # Cosine reward function parameters
    cosine_min_value_wrong: float = field(default=-1.0)
    cosine_max_value_wrong: float = field(default=-0.5)
    cosine_min_value_correct: float = field(default=0.5)
    cosine_max_value_correct: float = field(default=1.0)
    cosine_max_len: int = field(default=1000)
    
    # Repetition penalty parameters
    repetition_n_grams: int = field(default=3)
    repetition_max_penalty: float = field(default=-0.5)
    
    # Code execution parameters
    parallel_code_exec_per_proc: int = field(default=2)
    code_provider: str = field(default="e2b")
    code_eval_test_batch_size: int = field(default=1)
    code_eval_scoring_mode: str = field(default="weighted_sum")
    code_language: str = field(default="python")
    
    # Length punishment parameters
    max_completion_len: int = field(default=2048)
    soft_punish_cache: int = field(default=100)


@dataclass
class SimpleModelArgs:
    """Simple model arguments from YAML"""
    model_name_or_path: str
    lora_r: int = 16
    lora_alpha: int = 32
    torch_dtype: str = "auto"
    attn_implementation: str = "flash_attention_2"


def create_simple_args_from_yaml(yaml_config: dict):
    """Create simple argument objects from YAML"""
    
    # Required parameters
    if 'dataset_name' not in yaml_config:
        logger.error("❌ 'dataset_name' is required in YAML config")
        sys.exit(1)
    
    if 'model_name_or_path' not in yaml_config:
        logger.error("❌ 'model_name_or_path' is required in YAML config")
        sys.exit(1)
    
    # Create script args
    script_args = SimpleScriptArgs(
        dataset_name=yaml_config['dataset_name'],
        dataset_config=yaml_config.get('dataset_config', 'main'),
        dataset_train_split=yaml_config.get('dataset_train_split', 'train'),
        dataset_test_split=yaml_config.get('dataset_test_split', 'test'),
        dataset_num_proc=yaml_config.get('dataset_num_proc', 4),
        reward_funcs=yaml_config.get('reward_funcs', ["accuracy", "format", "reasoning_steps"])
    )
    
    # Create model args
    model_args = SimpleModelArgs(
        model_name_or_path=yaml_config['model_name_or_path'],
        lora_r=yaml_config.get('lora_r', 16),
        lora_alpha=yaml_config.get('lora_alpha', 32),
        torch_dtype=yaml_config.get('torch_dtype', 'auto'),
        attn_implementation=yaml_config.get('attn_implementation', 'flash_attention_2')
    )
    
    return script_args, model_args


# Patched generate function for GRPO compatibility
def patched_unsloth_fast_generate(self, *args, **kwargs):
    """Patched version of unsloth's generate function for GRPO compatibility"""
    FastLlamaModel.for_inference(self)
    dtype = _get_dtype(self.config.torch_dtype)

    if hasattr(self, "config") and hasattr(self.config, "max_position_embeddings"):
        if "input_ids" in kwargs and kwargs["input_ids"] is not None and "max_new_tokens" in kwargs:
            if kwargs["input_ids"].shape[-1] + kwargs["max_new_tokens"] > self.config.max_position_embeddings:
                raise ValueError(
                    f'Unsloth: input length {kwargs["input_ids"].shape[-1]} + max_new_tokens {kwargs["max_new_tokens"]} exceeds the maximum sequence length of {self.config.max_position_embeddings}!\n'
                    'You will need to do long context extension by increasing the `max_seq_length` in `FastLanguageModel.from_pretrained`.'
                )

    kwargs["cache_implementation"] = "dynamic"
    kwargs.pop("token_type_ids", None)

    model_eos_token_id = getattr(self.config, "eos_token_id", None)
    if model_eos_token_id is not None and hasattr(model_eos_token_id, "__iter__"):
        model_eos_token_id = model_eos_token_id[0]

    kwargs["pad_token_id"] = kwargs.pop("pad_token_id", model_eos_token_id)

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
        output = self._old_generate(*args, **kwargs)

    FastLlamaModel.for_training(self)
    return output


def get_unsloth_model(model_args: SimpleModelArgs, training_args: GRPOConfig):
    """Initialize Unsloth model with optimizations for GRPO training"""
    logger.info("🔧 Initializing Unsloth model for GRPO training...")
    max_seq_length = getattr(training_args, "max_length", 8192)
    lora_rank = model_args.lora_r
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_args.model_name_or_path,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=False,
        max_lora_rank=lora_rank,
    )
    
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=target_modules,
        lora_alpha=lora_rank * 2,
        use_gradient_checkpointing="unsloth" if training_args.gradient_checkpointing else None,
        random_state=training_args.seed,
    )
    
    # Apply generate function patch
    logger.info("🔧 Applying generate function patch for GRPO compatibility...")
    if hasattr(model, "_old_generate") and model.generate.__name__ == 'unsloth_fast_generate':
        model.generate = types.MethodType(patched_unsloth_fast_generate, model)
    elif model.generate.__name__ != 'patched_unsloth_fast_generate':
        if not hasattr(model, "_original_hf_generate"):
            model._original_hf_generate = model.generate
        model.generate = types.MethodType(patched_unsloth_fast_generate, model)
    
    return model, tokenizer


def prepare_dataset(script_args: SimpleScriptArgs):
    """Prepare dataset for GRPO training - returns either Dataset or DatasetDict"""
    logger.info(f"📊 Loading dataset: {script_args.dataset_name}")
    

    try:
        dataset = load_dataset_from_source(
            script_args.dataset_name, 
            name=script_args.dataset_config
        )
        dataset = dataset.map(make_conversation)
        
        for split in dataset:
            if "messages" in dataset[split].column_names:
                dataset[split] = dataset[split].remove_columns("messages")
        # Ensure dataset is a DatasetDict
        total_samples = sum(len(dataset[split]) for split in dataset.keys())
        logger.info(f"✅ Dataset prepared using open-r1: {total_samples} total samples")
        logger.info(f"   Available splits: {list(dataset.keys())}")
        return dataset
        
    except Exception as e:
        logger.warning(f"⚠️ open-r1 dataset loading failed: {e}")
        logger.info("🔄 Falling back to basic dataset loading...")
    
    # Fallback implementation
    if script_args.dataset_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", script_args.dataset_config)
        
        def format_gsm8k(examples):
            formatted = []
            for question, answer in zip(examples["question"], examples["answer"]):
                if "####" in answer:
                    final_answer = answer.split("####")[1].strip()
                else:
                    final_answer = "Unknown"
                
                formatted.append({
                    "prompt": [
                        {"role": "system", "content": "Solve this math problem step by step. Show your reasoning and provide the final answer."},
                        {"role": "user", "content": question}
                    ],
                    "answer": final_answer
                })
            
            return {"formatted": formatted}
        
        dataset = dataset.map(format_gsm8k, batched=True, remove_columns=dataset["train"].column_names)
        formatted_data = []
        for split_data in dataset.values():
            for item in split_data:
                formatted_data.extend(item["formatted"])
        
        dataset = Dataset.from_list(formatted_data)
    else:
        # For custom datasets like JSONL files
        if script_args.dataset_name.endswith('.jsonl'):
            dataset = Dataset.from_json(script_args.dataset_name)
            logger.info(f"✅ Loaded custom JSONL dataset: {len(dataset)} samples")
        else:
            raise ValueError(f"Dataset {script_args.dataset_name} not supported without open-r1")
    
    logger.info(f"✅ Dataset prepared: {len(dataset)} samples")
    return dataset


def get_reward_functions(script_args: SimpleScriptArgs) -> List[callable]:
    """Get reward functions for GRPO training using open-r1's professional implementations"""
    
    if OPEN_R1_REWARDS_AVAILABLE:
        try:
            # Ensure script_args has all required attributes for professional reward functions
            required_attrs = {
                'cosine_min_value_wrong': -1.0,
                'cosine_max_value_wrong': -0.5,
                'cosine_min_value_correct': 0.5,
                'cosine_max_value_correct': 1.0,
                'cosine_max_len': 1000,
                'repetition_n_grams': 3,
                'repetition_max_penalty': -0.5,
                'parallel_code_exec_per_proc': 2,
                'code_provider': 'e2b',
                'code_eval_test_batch_size': 1,
                'code_eval_scoring_mode': 'weighted_sum',
                'code_language': 'python',
                'max_completion_len': 2048,
                'soft_punish_cache': 100
            }
            
            for attr, default_value in required_attrs.items():
                if not hasattr(script_args, attr):
                    setattr(script_args, attr, default_value)
                    logger.debug(f"   Added missing attribute: {attr} = {default_value}")
            
            # Use the professional get_reward_funcs from open_r1.rewards
            reward_funcs = get_reward_funcs(script_args)
            logger.info(f"✅ Using open-r1 professional reward functions: {len(reward_funcs)} functions loaded")
            logger.info(f"   Configured functions: {getattr(script_args, 'reward_funcs', 'default')}")
            
            # Log function details
            func_names = getattr(script_args, 'reward_funcs', [])
            func_details = []
            for i, func in enumerate(reward_funcs):
                func_name = func_names[i] if i < len(func_names) else f"func_{i}"
                func_details.append(f"{func_name}({func.__name__})")
            logger.info(f"   Function details: {', '.join(func_details)}")
            
            return reward_funcs
        except Exception as e:
            logger.warning(f"⚠️ open-r1 reward functions failed to load: {e}")
            logger.info("🔄 Falling back to basic reward functions...")
            import traceback
            logger.debug(f"   Error details: {traceback.format_exc()}")
    
    # Fallback: open-r1 is not available
    logger.error("❌ open-r1 is not available - cannot load professional reward functions")
    logger.error("💡 Please install open-r1 and its dependencies for GRPO training")
    logger.error("   pip install latex2sympy2_extended math_verify")
    raise ImportError("open-r1 reward functions are required for GRPO training but not available")


def _log_rank0(message, level="info"):
    """Helper function to log only on rank 0"""
    if USE_DDP_LOGGER:
        if level == "info":
            logger.info_rank0(message)
        elif level == "warning":
            logger.warning_rank0(message)
        elif level == "error":
            logger.error_rank0(message)
    else:
        getattr(logger, level)(message)

def main():
    """Main training function - Pure YAML configuration"""
    _log_rank0("🚀 Starting GRPO Multi-GPU Training (Pure YAML Mode)")
    
    # Find YAML configuration file
    config_file = None
    
    # Check common locations (prioritize fixed config)
    for potential_config in ['configs/grpo_open_r1_config.yaml', 'config/config_unsloth.yaml']:
        if os.path.exists(potential_config):
            config_file = potential_config
            _log_rank0(f"🔍 Found config file: {config_file}")
            break
    
    if not config_file:
        _log_rank0("❌ No YAML configuration file found!", "error")
        _log_rank0("Please ensure one of these files exists:", "error")
        _log_rank0("  - config/config_unsloth.yaml", "error")
        sys.exit(1)
    
    # Load YAML configuration
    yaml_config = load_yaml_config(config_file)
    
    # Create arguments from YAML
    script_args, model_args = create_simple_args_from_yaml(yaml_config)
    training_args = create_grpo_config_from_yaml(yaml_config)
    
    # Set seed for reproducibility
    set_seed(training_args.seed)
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    
    # Log process info
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    
    _log_rank0(f"📊 Dataset: {script_args.dataset_name}")
    _log_rank0(f"🤖 Model: {model_args.model_name_or_path}")
    _log_rank0(f"🎯 Reward functions: {script_args.reward_funcs}")
    _log_rank0(f"📁 Output dir: {training_args.output_dir}")
    _log_rank0(f"🔢 Batch size per device: {training_args.per_device_train_batch_size}")
    _log_rank0(f"📈 Learning rate: {training_args.learning_rate}")

    # Enable unsloth-multigpu
    _log_rank0("🚀 Enabling unsloth-multigpu support...")
    
    grpo_success = ump.enable_grpo_support()
    if grpo_success:
        _log_rank0("✅ GRPO hooks enabled successfully")
    else:
        _log_rank0("❌ GRPO hooks failed to enable", "error")
        return

    try:
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        batch_size_per_gpu = max(1, training_args.per_device_train_batch_size)
        
        ump.enable_multi_gpu(
            num_gpus=world_size if world_size > 1 else 2,
            batch_size_per_gpu=batch_size_per_gpu,
            ddp_backend="nccl",
            enable_memory_optimization=True,
            enable_gradient_checkpointing=training_args.gradient_checkpointing
        )
        _log_rank0("✅ Multi-GPU support enabled")
    except Exception as e:
        _log_rank0(f"⚠️ Multi-GPU setup failed: {e}", "warning")
        _log_rank0("🔄 Continuing in single GPU mode")

    # Load dataset
    dataset = prepare_dataset(script_args)
    
    # Handle dataset splits - support both Dataset and DatasetDict
    if hasattr(dataset, 'keys') and callable(dataset.keys):
        # DatasetDict case (from open-r1)
        _log_rank0("📊 Processing DatasetDict from open-r1")
        
        # Use configured splits or defaults
        train_split = getattr(script_args, 'dataset_train_split', 'train')
        test_split = getattr(script_args, 'dataset_test_split', 'test')
        
        # Get train dataset
        if train_split in dataset:
            train_dataset = dataset[train_split]
            _log_rank0(f"✅ Using '{train_split}' split for training: {len(train_dataset)} samples")
        else:
            # Fallback to first available split
            available_splits = list(dataset.keys())
            train_split = available_splits[0]
            train_dataset = dataset[train_split]
            _log_rank0(f"⚠️ Train split '{script_args.dataset_train_split}' not found, using '{train_split}': {len(train_dataset)} samples", "warning")
        
        # Get eval dataset
        if training_args.eval_strategy != "no":
            if test_split in dataset:
                eval_dataset = dataset[test_split]
                _log_rank0(f"✅ Using '{test_split}' split for evaluation: {len(eval_dataset)} samples")
            elif len(list(dataset.keys())) > 1:
                # Use a different split for eval
                eval_split = [s for s in dataset.keys() if s != train_split][0]
                eval_dataset = dataset[eval_split]
                _log_rank0(f"✅ Using '{eval_split}' split for evaluation: {len(eval_dataset)} samples")
            else:
                # Split the train dataset
                if len(train_dataset) > 100:
                    train_size = int(0.9 * len(train_dataset))
                    dataset_split = train_dataset.train_test_split(train_size=train_size, seed=training_args.seed)
                    train_dataset = dataset_split["train"]
                    eval_dataset = dataset_split["test"]
                    _log_rank0(f"✅ Split single dataset: train={len(train_dataset)}, eval={len(eval_dataset)}")
                else:
                    eval_dataset = None
                    _log_rank0("📊 Dataset too small for evaluation split")
        else:
            eval_dataset = None
            _log_rank0("🔄 Evaluation disabled by configuration")
            
    else:
        # Single Dataset case (fallback path)
        _log_rank0("📊 Processing single Dataset")
        if len(dataset) > 100:
            train_size = int(0.9 * len(dataset))
            dataset_split = dataset.train_test_split(train_size=train_size, seed=training_args.seed)
            train_dataset = dataset_split["train"]
            eval_dataset = dataset_split["test"] if training_args.eval_strategy != "no" else None
        else:
            train_dataset = dataset
            eval_dataset = None
    
    _log_rank0(f"📊 Train dataset size: {len(train_dataset)}")
    if eval_dataset:
        _log_rank0(f"📊 Eval dataset size: {len(eval_dataset)}")

    # Load model and tokenizer
    model, tokenizer = get_unsloth_model(model_args, training_args)
    _log_rank0("✅ Unsloth model and tokenizer loaded")

    # Get reward functions
    reward_funcs = get_reward_functions(script_args)

    # Adjust for multi-GPU
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        world_size = int(os.environ['WORLD_SIZE'])
        training_args.num_generations = max(4, training_args.num_generations // world_size)
        _log_rank0(f"🔧 Multi-GPU detected: adjusted generations to {training_args.num_generations} per GPU")

    # Get callbacks if available (disabled for now due to compatibility issues)
    callbacks = None
    # Disable open-r1 callbacks for now as they cause issues with GRPOConfig
    # if OPEN_R1_AVAILABLE:
    #     try:
    #         callbacks = get_callbacks(training_args, model_args)
    #         logger.info("✅ Using open-r1 callbacks")
    #     except Exception as e:
    #         logger.warning(f"⚠️ Failed to get open-r1 callbacks: {e}")
    _log_rank0("🔧 Using default callbacks (open-r1 callbacks disabled for compatibility)")

    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
        processing_class=tokenizer,
    )
    _log_rank0("✅ GRPO trainer initialized")

    # Display status
    _log_rank0("\n📋 System Status:")
    status = ump.get_multi_gpu_status()
    _log_rank0(f"   Multi-GPU enabled: {status['enabled']}")
    if status['enabled']:
        _log_rank0(f"   Number of GPUs: {status['num_gpus']}")
    
    grpo_status = ump.get_grpo_status()
    _log_rank0(f"   GRPO support: {grpo_status.get('grpo_hooks_applied', False)}")

    # Training loop
    _log_rank0("🚀 Starting GRPO training...")
    _log_rank0("💡 Watch the 'reward' columns to see learning progress!")
    
    try:
        train_result = trainer.train()
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
        _log_rank0("✅ GRPO training completed successfully!")

    except Exception as e:
        _log_rank0(f"❌ Training failed: {e}", "error")
        raise
    finally:
        # Clean up unsloth-multigpu
        _log_rank0("🧹 Cleaning up unsloth-multigpu...")
        ump.disable_grpo_support()
        ump.disable_multi_gpu()

    # Save model
    _log_rank0("💾 Saving trained model...")
    trainer.save_model(training_args.output_dir)
    _log_rank0(f"✅ Model saved to {training_args.output_dir}")

    # Save additional artifacts on main process
    if trainer.accelerator.is_main_process:
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)
        
        kwargs = {
            "dataset_name": script_args.dataset_name,
            "tags": ["unsloth-multigpu", "grpo", "math-reasoning"],
        }
        trainer.create_model_card(**kwargs)

    # Evaluate
    if training_args.do_eval and eval_dataset is not None:
        _log_rank0("📊 Running evaluation...")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        _log_rank0("✅ Evaluation completed")

    _log_rank0("🎉 GRPO training pipeline completed successfully!")


if __name__ == "__main__":
    main()