"""
Configuration management tools
Provide loading, validation, saving, and management of multi-GPU training configurations
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import yaml

logger = logging.getLogger(__name__)


@dataclass
class MultiGPUConfig:
    """Multi-GPU training configuration"""
    
    # Basic configuration
    enabled: bool = True
    num_gpus: int = None  # None means auto-detection
    device_ids: List[int] = field(default_factory=list)  # Empty list means auto-selection
    
    # Training configuration
    batch_size_per_gpu: int = 2
    gradient_accumulation_steps: int = 1
    max_steps: int = 1000
    num_train_epochs: float = 1.0
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    
    # DDP configuration
    enable_gradient_checkpointing: bool = True
    enable_mixed_precision: bool = True
    ddp_backend: str = "nccl"  # nccl, gloo, or mpi
    ddp_timeout_minutes: int = 30  # DDP operation timeout
    find_unused_parameters: bool = False  # DDP find_unused_parameters
    
    # Memory management
    memory_optimization: bool = True
    memory_warning_threshold: float = 0.85
    memory_critical_threshold: float = 0.95
    enable_auto_cleanup: bool = True
    cleanup_interval: float = 10.0
    
    # Performance optimization
    enable_cudnn_benchmark: bool = True
    enable_tf32: bool = True
    pin_memory: bool = True
    num_workers: int = 4
    
    # Monitoring and logging
    log_level: str = "INFO"
    log_dir: str = "logs"
    enable_wandb: bool = False
    wandb_project: str = "unsloth-multigpu"
    save_steps: int = 500
    eval_steps: int = 100
    
    # Fault tolerance and stability
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_fault_tolerance: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self)
    
    def __post_init__(self):
        """Post-initialization processing"""
        # Automatically detect GPU number
        if self.num_gpus is None:
            if torch.cuda.is_available():
                self.num_gpus = torch.cuda.device_count()
            else:
                self.num_gpus = 0
                self.enabled = False
        
        # Automatically select devices
        if not self.device_ids and self.num_gpus > 0:
            self.device_ids = list(range(min(self.num_gpus, torch.cuda.device_count())))
        
        # Validate configuration
        self._validate()
    
    def _validate(self):
        """Validate configuration validity"""
        if self.enabled and self.num_gpus < 2:
            logger.warning("⚠️ Multi-GPU mode requires at least 2 GPUs, automatically disabling multi-GPU")
            self.enabled = False
        
        if self.ddp_backend not in ["nccl", "gloo", "mpi"]:
            raise ValueError(f"Invalid DDP backend: {self.ddp_backend}")
        
        if self.num_gpus > 1 and not torch.cuda.is_available():
            raise ValueError("CUDA is required for multi-GPU training")
        
        if not 0 < self.memory_warning_threshold < self.memory_critical_threshold < 1:
            raise ValueError("Invalid memory threshold settings")
        
        if self.device_ids:
            available_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
            invalid_devices = [d for d in self.device_ids if d not in available_devices]
            if invalid_devices:
                raise ValueError(f"Invalid device IDs: {invalid_devices}")


class ConfigManager:
    """Configuration manager"""
    
    def __init__(self, config_dir: str = "configs"):
        """
        Initialize configuration manager
        
        Args:
            config_dir: Configuration file directory
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        self._config_cache = {}
        logger.info(f"🔧 Configuration manager initialized: {self.config_dir}")
    
    def save_config(self, config: MultiGPUConfig, filename: str = "multi_gpu_config") -> str:
        """
        Save configuration to file
        
        Args:
            config: Configuration object
            filename: File name (without extension)
            
        Returns:
            str: Saved file path
        """
        config_dict = asdict(config)
        
        # Save as JSON format
        json_path = self.config_dir / f"{filename}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        # Save as YAML format
        yaml_path = self.config_dir / f"{filename}.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"💾 Config saved: {json_path} and {yaml_path}")
        return str(json_path)
    
    def load_config(self, filename: str) -> MultiGPUConfig:
        """
        Load configuration from file
        
        Args:
            filename: File name or path
            
        Returns:
            MultiGPUConfig: Configuration object
        """
        # Check cache
        if filename in self._config_cache:
            logger.debug(f"📋 Loaded config from cache: {filename}")
            return self._config_cache[filename]
        
        # Determine file path
        if not os.path.isabs(filename):
            # Try different extensions
            for ext in ['.json', '.yaml', '.yml', '']:
                test_path = self.config_dir / f"{filename}{ext}"
                if test_path.exists():
                    filepath = test_path
                    break
            else:
                raise FileNotFoundError(f"Config file not found: {filename}")
        else:
            filepath = Path(filename)
        
        # Load config
        with open(filepath, 'r', encoding='utf-8') as f:
            if filepath.suffix.lower() in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            else:
                config_dict = json.load(f)
        
        # Create config object
        config = MultiGPUConfig(**config_dict)
        
        # Cache config
        self._config_cache[filename] = config
        
        logger.info(f"📁 Config loaded: {filepath}")
        return config
    
    def create_default_configs(self):
        """Create default configuration files"""
        configs = {
            "default": MultiGPUConfig(),
            "high_performance": MultiGPUConfig(
                batch_size_per_gpu=4,
                enable_mixed_precision=True,
                enable_gradient_checkpointing=True,
                gradient_aggregation="mean",
                memory_optimization=True
            ),
            "memory_efficient": MultiGPUConfig(
                batch_size_per_gpu=1,
                gradient_accumulation_steps=4,
                enable_gradient_checkpointing=True,
                memory_warning_threshold=0.8,
                memory_critical_threshold=0.9,
                cleanup_interval=5.0
            ),
            "debug": MultiGPUConfig(
                batch_size_per_gpu=1,
                max_steps=10,
                log_level="DEBUG",
                save_steps=5,
                eval_steps=5
            )
        }
        
        for name, config in configs.items():
            self.save_config(config, f"{name}_config")
        
        logger.info(f"✅ Default configuration files created: {len(configs)}")
    
    def validate_config(self, config: MultiGPUConfig) -> Dict[str, Any]:
        """
        Validate the integrity and reasonableness of the configuration
        
        Args:
            config: Configuration object
            
        Returns:
            Dict: Validation result
        """
        result = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'suggestions': []
        }
        
        try:
            # Basic validation (already done in __post_init__)
            config._validate()
            
            # Performance-related validation
            if config.batch_size_per_gpu * config.num_gpus > 32:
                result['warnings'].append(
                    f"Total batch size {config.batch_size_per_gpu * config.num_gpus} may be too large"
                )
            
            if config.learning_rate > 1e-3:
                result['warnings'].append(f"Learning rate {config.learning_rate} may be too high")
            
            # Memory-related validation
            if not config.memory_optimization and config.batch_size_per_gpu > 2:
                result['suggestions'].append("It is recommended to enable memory optimization to support larger batch sizes")
            
            # Device-related validation
            if torch.cuda.is_available():
                available_memory = []
                for device_id in config.device_ids:
                    if device_id < torch.cuda.device_count():
                        props = torch.cuda.get_device_properties(device_id)
                        available_memory.append(props.total_memory / 1024**3)  # GB
                
                min_memory = min(available_memory) if available_memory else 0
                if min_memory < 8:  # 8GB
                    result['warnings'].append(f"Device memory is low ({min_memory:.1f}GB), consider reducing batch size")
            
            # Training config validation
            if config.warmup_steps > config.max_steps * 0.2:
                result['warnings'].append("Warmup steps may be too many")
            
            if config.gradient_accumulation_steps > 8:
                result['suggestions'].append("Gradient accumulation steps are high, consider increasing actual batch size")
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(str(e))
        
        return result
    
    def get_optimal_config(self, 
                          target_batch_size: int = None,
                          available_memory_gb: float = None,
                          training_type: str = "standard") -> MultiGPUConfig:
        """
        Get optimal configuration
        
        Args:
            target_batch_size: Target total batch size
            available_memory_gb: Available memory (GB)
            training_type: Training type (standard, fast, memory_efficient)
            
        Returns:
            MultiGPUConfig: Optimized configuration
        """
        # Base config
        if training_type == "fast":
            base_config = MultiGPUConfig(
                enable_mixed_precision=True,
                enable_gradient_checkpointing=False,
                gradient_aggregation="mean",
                memory_optimization=False
            )
        elif training_type == "memory_efficient":
            base_config = MultiGPUConfig(
                batch_size_per_gpu=1,
                gradient_accumulation_steps=4,
                enable_gradient_checkpointing=True,
                memory_optimization=True,
                memory_warning_threshold=0.75
            )
        else:  # standard
            base_config = MultiGPUConfig()
        
        # Adjust according to target batch size
        if target_batch_size:
            num_gpus = base_config.num_gpus if base_config.num_gpus > 0 else 1
            batch_per_gpu = max(1, target_batch_size // num_gpus)
            
            if batch_per_gpu * num_gpus < target_batch_size:
                # Use gradient accumulation to make up
                remaining = target_batch_size - (batch_per_gpu * num_gpus)
                base_config.gradient_accumulation_steps = max(1, remaining // num_gpus + 1)
            
            base_config.batch_size_per_gpu = batch_per_gpu
        
        # Adjust according to available memory
        if available_memory_gb:
            if available_memory_gb < 8:
                base_config.batch_size_per_gpu = 1
                base_config.enable_gradient_checkpointing = True
                base_config.memory_optimization = True
            elif available_memory_gb < 16:
                base_config.batch_size_per_gpu = min(2, base_config.batch_size_per_gpu)
            # else: use default config
        
        logger.info(f"🎯 Generated optimal config: {training_type} type")
        return base_config
    
    def compare_configs(self, config1: MultiGPUConfig, config2: MultiGPUConfig) -> Dict[str, Any]:
        """
        Compare two configurations
        
        Args:
            config1: Config 1
            config2: Config 2
            
        Returns:
            Dict: Comparison result
        """
        dict1 = asdict(config1)
        dict2 = asdict(config2)
        
        differences = {}
        for key in dict1:
            if dict1[key] != dict2[key]:
                differences[key] = {
                    'config1': dict1[key],
                    'config2': dict2[key]
                }
        
        return {
            'identical': len(differences) == 0,
            'differences': differences,
            'difference_count': len(differences)
        }
    
    def export_config_template(self, filename: str = "config_template"):
        """
        Export configuration template
        
        Args:
            filename: Template file name
        """
        template = {
            "# Multi-GPU Training Configuration Template": None,
            "enabled": "bool - Whether to enable multi-GPU training",
            "num_gpus": "int or null - Number of GPUs, null for auto-detection",
            "device_ids": "list - List of GPU device IDs, empty list for auto-selection",
            "batch_size_per_gpu": "int - Batch size per GPU",
            "gradient_aggregation": "str - Gradient aggregation method (mean/sum/weighted_mean/median)",
            "batch_sharding_strategy": "str - Batch sharding strategy (uniform/adaptive/weighted)",
            "enable_gradient_checkpointing": "bool - Whether to enable gradient checkpointing",
            "enable_mixed_precision": "bool - Whether to enable mixed precision",
            "memory_optimization": "bool - Whether to enable memory optimization",
            "log_level": "str - Log level (DEBUG/INFO/WARNING/ERROR)",
            "# For more configuration options, see the MultiGPUConfig class": None
        }
        
        template_path = self.config_dir / f"{filename}.yaml"
        with open(template_path, 'w', encoding='utf-8') as f:
            yaml.dump(template, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"📋 Configuration template exported: {template_path}")
    
    def list_configs(self) -> List[str]:
        """List all available configurations"""
        configs = []
        for ext in ['.json', '.yaml', '.yml']:
            configs.extend([
                f.stem for f in self.config_dir.glob(f"*{ext}")
            ])
        return sorted(set(configs))


def load_config_from_file(filepath: str) -> MultiGPUConfig:
    """
    Convenience function to load configuration from file
    
    Args:
        filepath: Configuration file path
        
    Returns:
        MultiGPUConfig: Configuration object
    """
    manager = ConfigManager()
    return manager.load_config(filepath)


def create_config_from_dict(config_dict: Dict[str, Any]) -> MultiGPUConfig:
    """
    Convenience function to create configuration from dict
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        MultiGPUConfig: Configuration object
    """
    return MultiGPUConfig(**config_dict)


def get_default_config() -> MultiGPUConfig:
    """Get default configuration"""
    return MultiGPUConfig()


def setup_config_for_environment() -> MultiGPUConfig:
    """
    Automatically set up configuration based on current environment
    
    Returns:
        MultiGPUConfig: Configuration suitable for current environment
    """
    config = MultiGPUConfig()
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        total_memory = 0
        
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            total_memory += props.total_memory / 1024**3  # GB
        
        avg_memory = total_memory / device_count if device_count > 0 else 0
        
        # Adjust config based on average memory
        if avg_memory < 8:
            config.batch_size_per_gpu = 1
            config.enable_gradient_checkpointing = True
            config.memory_optimization = True
        elif avg_memory < 16:
            config.batch_size_per_gpu = 2
        else:
            config.batch_size_per_gpu = 4
        
        logger.info(f"🔍 Environment detected: {device_count} GPU(s), average memory {avg_memory:.1f}GB")
        logger.info(f"🎯 Recommended batch size: {config.batch_size_per_gpu} per GPU")
    
    return config 