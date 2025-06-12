"""
Memory Management Module
Responsible for memory safety checks, monitoring, optimization, and OOM prevention during multi-GPU training
"""

import gc
import logging
import os
import sys
import time
import warnings
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import psutil
import torch

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Memory Manager
    
    Responsible:
    1. GPU and CPU memory monitoring
    2. Memory safety checks and warnings
    3. Dynamic memory optimization
    4. OOM prevention mechanism
    5. Memory fragmentation cleanup
    """
    
    def __init__(self, 
                 warning_threshold: float = 0.85,
                 critical_threshold: float = 0.95,
                 enable_auto_cleanup: bool = True,
                 cleanup_interval: float = 10.0):
        """
        Initialize Memory Manager
        
        Args:
            warning_threshold: Memory usage warning threshold
            critical_threshold: Memory usage danger threshold 
            enable_auto_cleanup: Whether to enable automatic cleanup
            cleanup_interval: Automatic cleanup interval (seconds)
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.enable_auto_cleanup = enable_auto_cleanup
        self.cleanup_interval = cleanup_interval
        
        # Memory statistics
        self.memory_stats = {
            'peak_gpu_memory': {},
            'peak_cpu_memory': 0,
            'oom_count': 0,
            'cleanup_count': 0,
            'warning_count': 0
        }
        
        # Automatic cleanup status
        self.last_cleanup_time = time.time()
        
        logger.info("🧠 Memory Manager initialization completed")
        logger.info(f"📊 Warning threshold: {warning_threshold*100:.1f}%, Danger threshold: {critical_threshold*100:.1f}%")
    
    def get_gpu_memory_info(self, device_id: int = None) -> Dict[str, Any]:
        """
        Get GPU memory information
        
        Args:
            device_id: GPU device ID, None for current device
            
        Returns:
            Dict: GPU memory information
        """
        if not torch.cuda.is_available():
            return {'error': 'CUDA not available'}
        
        if device_id is None:
            device_id = torch.cuda.current_device()
        
        try:
            # Get memory information
            total_memory = torch.cuda.get_device_properties(device_id).total_memory
            allocated_memory = torch.cuda.memory_allocated(device_id)
            cached_memory = torch.cuda.memory_reserved(device_id)
            free_memory = total_memory - cached_memory
            
            utilization = allocated_memory / total_memory if total_memory > 0 else 0
            cache_utilization = cached_memory / total_memory if total_memory > 0 else 0
            
            info = {
                'device_id': device_id,
                'device_name': torch.cuda.get_device_name(device_id),
                'total_memory_mb': total_memory / 1024 / 1024,
                'allocated_memory_mb': allocated_memory / 1024 / 1024,
                'cached_memory_mb': cached_memory / 1024 / 1024,
                'free_memory_mb': free_memory / 1024 / 1024,
                'utilization': utilization,
                'cache_utilization': cache_utilization,
                'memory_efficiency': allocated_memory / cached_memory if cached_memory > 0 else 0
            }
            
            # Update peak statistics
            current_peak = self.memory_stats['peak_gpu_memory'].get(device_id, 0)
            if allocated_memory > current_peak:
                self.memory_stats['peak_gpu_memory'][device_id] = allocated_memory
            
            return info
            
        except Exception as e:
            logger.error(f"❌ Failed to get GPU {device_id} memory information: {e}")
            return {'error': str(e)}
    
    def get_cpu_memory_info(self) -> Dict[str, Any]:
        """
        Get CPU memory information
        
        Returns:
            Dict: CPU memory information
        """
        try:
            # System memory information
            memory = psutil.virtual_memory()
            
            # Current process memory information
            process = psutil.Process()
            process_memory = process.memory_info()
            
            info = {
                'total_memory_mb': memory.total / 1024 / 1024,
                'available_memory_mb': memory.available / 1024 / 1024,
                'used_memory_mb': memory.used / 1024 / 1024,
                'utilization': memory.percent / 100,
                'process_rss_mb': process_memory.rss / 1024 / 1024,
                'process_vms_mb': process_memory.vms / 1024 / 1024,
                'process_percent': process.memory_percent()
            }
            
            # Update peak statistics
            if process_memory.rss > self.memory_stats['peak_cpu_memory']:
                self.memory_stats['peak_cpu_memory'] = process_memory.rss
            
            return info
            
        except Exception as e:
            logger.error(f"❌ Failed to get CPU memory information: {e}")
            return {'error': str(e)}
    
    def check_memory_safety(self, device_ids: List[int] = None) -> Dict[str, Any]:
        """
        Check memory safety status
        
        Args:
            device_ids: List of GPU devices to check
            
        Returns:
            Dict: Memory safety check results
        """
        results = {
            'safe': True,
            'warnings': [],
            'critical_issues': [],
            'gpu_status': {},
            'cpu_status': {},
            'recommendations': []
        }
        
        # Check CPU memory
        cpu_info = self.get_cpu_memory_info()
        if 'error' not in cpu_info:
            results['cpu_status'] = cpu_info
            
            if cpu_info['utilization'] > self.critical_threshold:
                results['safe'] = False
                results['critical_issues'].append(
                    f"CPU memory usage danger: {cpu_info['utilization']*100:.1f}% > {self.critical_threshold*100:.1f}%"
                )
            elif cpu_info['utilization'] > self.warning_threshold:
                results['warnings'].append(
                    f"CPU memory usage high: {cpu_info['utilization']*100:.1f}% > {self.warning_threshold*100:.1f}%"
                )
                self.memory_stats['warning_count'] += 1
        
        # Check GPU memory
        if torch.cuda.is_available():
            if device_ids is None:
                device_ids = list(range(torch.cuda.device_count()))
            
            for device_id in device_ids:
                gpu_info = self.get_gpu_memory_info(device_id)
                if 'error' not in gpu_info:
                    results['gpu_status'][device_id] = gpu_info
                    
                    if gpu_info['utilization'] > self.critical_threshold:
                        results['safe'] = False
                        results['critical_issues'].append(
                            f"GPU {device_id} memory usage danger: {gpu_info['utilization']*100:.1f}% > {self.critical_threshold*100:.1f}%"
                        )
                    elif gpu_info['utilization'] > self.warning_threshold:
                        results['warnings'].append(
                            f"GPU {device_id} memory usage high: {gpu_info['utilization']*100:.1f}% > {self.warning_threshold*100:.1f}%"
                        )
                        self.memory_stats['warning_count'] += 1
        
        # Generate recommendations
        if results['critical_issues']:
            results['recommendations'].extend([
                "Immediately release unnecessary variables and caches",
                "Reduce batch size or model size",
                "Enable gradient checkpointing to save memory",
                "Consider using CPU offloading technology"
            ])
        elif results['warnings']:
            results['recommendations'].extend([
                "Monitor memory usage trends",
                "Properly clean memory caches",
                "Optimize data loading and preprocessing"
            ])
        
        return results
    
    def cleanup_memory(self, device_ids: List[int] = None, aggressive: bool = False) -> Dict[str, Any]:
        """
        Clean memory
        
        Args:
            device_ids: List of GPU devices to clean
            aggressive: Whether to perform aggressive cleanup
            
        Returns:
            Dict: Cleanup results
        """
        results = {
            'success': True,
            'cleaned_memory_mb': 0,
            'gpu_cleanup': {},
            'cpu_cleanup': {},
            'errors': []
        }
        
        try:
            # CPU memory cleanup
            cpu_before = self.get_cpu_memory_info()
            
            # Python garbage collection
            collected = gc.collect()
            results['cpu_cleanup']['garbage_collected'] = collected
            
            cpu_after = self.get_cpu_memory_info()
            if 'error' not in cpu_before and 'error' not in cpu_after:
                cpu_freed = cpu_before['process_rss_mb'] - cpu_after['process_rss_mb']
                results['cpu_cleanup']['freed_memory_mb'] = max(0, cpu_freed)
                results['cleaned_memory_mb'] += max(0, cpu_freed)
            
            # GPU memory cleanup
            if torch.cuda.is_available():
                if device_ids is None:
                    device_ids = list(range(torch.cuda.device_count()))
                
                for device_id in device_ids:
                    try:
                        gpu_before = self.get_gpu_memory_info(device_id)
                        
                        with torch.cuda.device(device_id):
                            # Clear cache
                            torch.cuda.empty_cache()
                            
                            if aggressive:
                                # Aggressive cleanup: Force synchronization and multiple cleanups
                                torch.cuda.synchronize()
                                torch.cuda.empty_cache()
                                gc.collect()
                                torch.cuda.empty_cache()
                        
                        gpu_after = self.get_gpu_memory_info(device_id)
                        
                        if 'error' not in gpu_before and 'error' not in gpu_after:
                            gpu_freed = gpu_before['cached_memory_mb'] - gpu_after['cached_memory_mb']
                            results['gpu_cleanup'][device_id] = {
                                'freed_memory_mb': max(0, gpu_freed),
                                'before_utilization': gpu_before['utilization'],
                                'after_utilization': gpu_after['utilization']
                            }
                            results['cleaned_memory_mb'] += max(0, gpu_freed)
                        
                    except Exception as e:
                        error_msg = f"GPU {device_id} cleanup failed: {e}"
                        results['errors'].append(error_msg)
                        logger.error(f"❌ {error_msg}")
            
            self.memory_stats['cleanup_count'] += 1
            self.last_cleanup_time = time.time()
            
            if results['cleaned_memory_mb'] > 0:
                logger.info(f"🧹 Memory cleanup completed: Released {results['cleaned_memory_mb']:.1f} MB")
            else:
                logger.info("🧹 Memory cleanup completed: No memory to release")
                
        except Exception as e:
            results['success'] = False
            results['errors'].append(str(e))
            logger.error(f"❌ Memory cleanup failed: {e}")
        
        return results
    
    def prevent_oom(self, required_memory_mb: float, device_id: int = None) -> bool:
        """
        OOM prevention check
        
        Args:
            required_memory_mb: Required memory size (MB)
            device_id: GPU device ID
            
        Returns:
            bool: Whether there is enough memory
        """
        try:
            if device_id is not None and torch.cuda.is_available():
                # GPU memory check
                gpu_info = self.get_gpu_memory_info(device_id)
                if 'error' in gpu_info:
                    return False
                
                available_memory = gpu_info['free_memory_mb']
                safety_margin = gpu_info['total_memory_mb'] * (1 - self.critical_threshold)
                
                if required_memory_mb > available_memory - safety_margin:
                    logger.warning(f"⚠️ GPU {device_id} may OOM: Need {required_memory_mb:.1f} MB, Available {available_memory:.1f} MB")
                    
                    # Try to clean memory
                    cleanup_result = self.cleanup_memory([device_id], aggressive=True)
                    
                    # Re-check
                    gpu_info_after = self.get_gpu_memory_info(device_id)
                    if 'error' not in gpu_info_after:
                        available_memory_after = gpu_info_after['free_memory_mb']
                        if required_memory_mb > available_memory_after - safety_margin:
                            self.memory_stats['oom_count'] += 1
                            return False
                
            else:
                # CPU memory check
                cpu_info = self.get_cpu_memory_info()
                if 'error' in cpu_info:
                    return False
                
                available_memory = cpu_info['available_memory_mb']
                safety_margin = cpu_info['total_memory_mb'] * (1 - self.critical_threshold)
                
                if required_memory_mb > available_memory - safety_margin:
                    logger.warning(f"⚠️ CPU may OOM: Need {required_memory_mb:.1f} MB, Available {available_memory:.1f} MB")
                    
                    # Try to clean memory
                    self.cleanup_memory(aggressive=True)
                    
                    # Re-check
                    cpu_info_after = self.get_cpu_memory_info()
                    if 'error' not in cpu_info_after:
                        available_memory_after = cpu_info_after['available_memory_mb']
                        if required_memory_mb > available_memory_after - safety_margin:
                            self.memory_stats['oom_count'] += 1
                            return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ OOM prevention check failed: {e}")
            return False
    
    @contextmanager
    def memory_guard(self, device_ids: List[int] = None, auto_cleanup: bool = True):
        """
        Memory protection context manager
        
        Args:
            device_ids: List of GPU devices to protect
            auto_cleanup: Whether to automatically clean up
        """
        # Record initial state
        initial_stats = {}
        if torch.cuda.is_available() and device_ids:
            for device_id in device_ids:
                initial_stats[device_id] = self.get_gpu_memory_info(device_id)
        
        initial_cpu = self.get_cpu_memory_info()
        
        try:
            yield self
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("💥 Detected OOM error, attempting memory cleanup...")
                self.memory_stats['oom_count'] += 1
                
                if auto_cleanup:
                    self.cleanup_memory(device_ids, aggressive=True)
                
            raise
            
        finally:
            # Check for memory leaks
            if auto_cleanup and self.enable_auto_cleanup:
                current_time = time.time()
                if current_time - self.last_cleanup_time > self.cleanup_interval:
                    self.cleanup_memory(device_ids)
    
    def optimize_memory_usage(self, model=None, enable_gradient_checkpointing: bool = True) -> Dict[str, Any]:
        """
        Optimize memory usage
        
        Args:
            model: Model to optimize
            enable_gradient_checkpointing: Whether to enable gradient checkpointing
            
        Returns:
            Dict: Optimization results
        """
        results = {
            'optimizations_applied': [],
            'memory_saved_mb': 0,
            'recommendations': []
        }
        
        try:
            # 1. Gradient checkpointing optimization
            if model is not None and enable_gradient_checkpointing:
                if hasattr(model, 'gradient_checkpointing_enable'):
                    model.gradient_checkpointing_enable()
                    results['optimizations_applied'].append('gradient_checkpointing')
                    results['memory_saved_mb'] += 500  # Estimated savings
                    logger.info("✅ Enabled gradient checkpointing")
            
            # 2. PyTorch optimization settings
            if torch.cuda.is_available():
                # Enable memory pool
                torch.cuda.empty_cache()
                results['optimizations_applied'].append('cuda_cache_optimization')
                
                # Optimize CUDA allocator
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
                results['optimizations_applied'].append('cuda_allocator_optimization')
            
            # 3. Data type optimization suggestions
            results['recommendations'].extend([
                "Use fp16 mixed precision training",
                "Properly reduce batch size",
                "Use LoRA or other parameter efficient fine-tuning methods",
                "Enable gradient accumulation to reduce memory peak"
            ])
            
            logger.info(f"🎯 Memory optimization completed: Applied {len(results['optimizations_applied'])} optimization items")
            
        except Exception as e:
            logger.error(f"❌ Memory optimization failed: {e}")
        
        return results
    
    def get_memory_report(self) -> Dict[str, Any]:
        """
        Get detailed memory report
        
        Returns:
            Dict: Memory report
        """
        report = {
            'timestamp': time.time(),
            'cpu_memory': self.get_cpu_memory_info(),
            'gpu_memory': {},
            'memory_stats': self.memory_stats.copy(),
            'system_info': {
                'python_version': sys.version,
                'torch_version': torch.__version__ if torch else 'Not Available',
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'Not Available'
            }
        }
        
        # Add memory information for all GPUs
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                report['gpu_memory'][i] = self.get_gpu_memory_info(i)
        
        return report
    
    def __del__(self):
        """Destructor, clean up resources"""
        try:
            if self.enable_auto_cleanup:
                self.cleanup_memory(aggressive=False)
        except Exception:
            pass 