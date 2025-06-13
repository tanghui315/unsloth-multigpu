"""
Logging system utilities
Provides structured logging, performance monitoring, and debugging features
"""

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional

import torch


class MultiGPULogger:
    """
    Multi-GPU training logger

    Features:
    1. Structured logging
    2. Performance metrics tracking
    3. GPU status monitoring
    4. Training process debugging
    """

    def __init__(self, 
                 name: str = "unsloth_multigpu",
                 level: int = logging.INFO,
                 log_dir: str = "logs",
                 max_file_size: int = 10*1024*1024,  # 10MB
                 backup_count: int = 5,
                 enable_console: bool = True,
                 enable_file: bool = True,
                 enable_json: bool = True):
        """
        Initialize the multi-GPU logger

        Args:
            name: Logger name
            level: Logging level
            log_dir: Log directory
            max_file_size: Maximum file size
            backup_count: Number of backup files
            enable_console: Enable console output
            enable_file: Enable file output
            enable_json: Enable JSON format
        """
        self.name = name
        self.level = level
        self.log_dir = log_dir
        self.enable_json = enable_json

        # Create log directory
        if enable_file:
            os.makedirs(log_dir, exist_ok=True)

        # Performance statistics
        self.performance_metrics = {
            'start_time': time.time(),
            'step_times': [],
            'gpu_memory_usage': {},
            'throughput_history': [],
            'error_count': 0,
            'warning_count': 0
        }

        # Set up logger
        self.logger = self._setup_logger(
            level, max_file_size, backup_count, 
            enable_console, enable_file
        )

        self.logger.info("🪵 Multi-GPU logging system initialized")

    def _setup_logger(self, level, max_file_size, backup_count, 
                     enable_console, enable_file):
        """Set up logger"""
        logger = logging.getLogger(self.name)
        logger.setLevel(level)

        # Clear existing handlers
        logger.handlers.clear()

        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = self._create_console_formatter()
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(level)
            logger.addHandler(console_handler)

        # File handler
        if enable_file:
            # Regular log file
            log_file = os.path.join(self.log_dir, f"{self.name}.log")
            file_handler = RotatingFileHandler(
                log_file, maxBytes=max_file_size, backupCount=backup_count
            )
            file_formatter = self._create_file_formatter()
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(level)
            logger.addHandler(file_handler)

            # JSON log file (if enabled)
            if self.enable_json:
                json_file = os.path.join(self.log_dir, f"{self.name}_structured.json")
                json_handler = RotatingFileHandler(
                    json_file, maxBytes=max_file_size, backupCount=backup_count
                )
                json_formatter = JSONFormatter()
                json_handler.setFormatter(json_formatter)
                json_handler.setLevel(level)
                logger.addHandler(json_handler)

        return logger

    def _create_console_formatter(self):
        """Create console formatter"""
        return ColoredFormatter(
            '%(asctime)s | %(levelname)8s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )

    def _create_file_formatter(self):
        """Create file formatter"""
        return logging.Formatter(
            '%(asctime)s | %(levelname)8s | %(name)s | %(filename)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def log_training_step(self, step: int, loss: float, metrics: Dict[str, Any] = None,
                         gpu_stats: Dict[int, Dict] = None):
        """
        Log training step

        Args:
            step: Step number
            loss: Loss value
            metrics: Other metrics
            gpu_stats: GPU statistics
        """
        step_time = time.time()
        self.performance_metrics['step_times'].append(step_time)

        # Calculate step interval
        if len(self.performance_metrics['step_times']) > 1:
            step_duration = step_time - self.performance_metrics['step_times'][-2]
        else:
            step_duration = 0

        # Basic info
        log_data = {
            'event': 'training_step',
            'step': step,
            'loss': loss,
            'step_duration': step_duration,
            'timestamp': step_time
        }

        # Add other metrics
        if metrics:
            log_data['metrics'] = metrics

        # Add GPU stats
        if gpu_stats:
            log_data['gpu_stats'] = gpu_stats
            self.performance_metrics['gpu_memory_usage'][step] = gpu_stats

        # Calculate throughput
        if step > 0 and step_duration > 0:
            throughput = 1.0 / step_duration
            log_data['throughput'] = throughput
            self.performance_metrics['throughput_history'].append(throughput)

        self.logger.info(f"📈 Step {step}: loss={loss:.4f}, duration={step_duration:.3f}s", 
                        extra={'structured_data': log_data})

    def log_gpu_status(self, device_stats: Dict[int, Dict]):
        """
        Log GPU status

        Args:
            device_stats: Device statistics
        """
        log_data = {
            'event': 'gpu_status',
            'device_stats': device_stats,
            'timestamp': time.time()
        }

        # Generate concise console info
        gpu_summary = []
        for device_id, stats in device_stats.items():
            if 'memory_allocated_mb' in stats and 'memory_reserved_mb' in stats:
                memory_usage = stats['memory_allocated_mb'] / stats['memory_reserved_mb'] * 100 if stats['memory_reserved_mb'] > 0 else 0
                gpu_summary.append(f"GPU{device_id}: {memory_usage:.1f}%")

        summary_text = ", ".join(gpu_summary)
        self.logger.info(f"💾 GPU status: {summary_text}", 
                        extra={'structured_data': log_data})

    def log_performance_summary(self):
        """Log performance summary"""
        current_time = time.time()
        total_duration = current_time - self.performance_metrics['start_time']

        # Calculate average metrics
        avg_throughput = 0
        if self.performance_metrics['throughput_history']:
            avg_throughput = sum(self.performance_metrics['throughput_history']) / len(self.performance_metrics['throughput_history'])

        total_steps = len(self.performance_metrics['step_times'])

        log_data = {
            'event': 'performance_summary',
            'total_duration': total_duration,
            'total_steps': total_steps,
            'avg_throughput': avg_throughput,
            'error_count': self.performance_metrics['error_count'],
            'warning_count': self.performance_metrics['warning_count'],
            'timestamp': current_time
        }

        self.logger.info(
            f"📊 Performance summary: {total_steps} steps, {total_duration:.1f}s, "
            f"Average throughput: {avg_throughput:.2f} steps/s",
            extra={'structured_data': log_data}
        )

    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """
        Log error

        Args:
            error: Exception object
            context: Error context
        """
        self.performance_metrics['error_count'] += 1

        log_data = {
            'event': 'error',
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': time.time()
        }

        if context:
            log_data['context'] = context

        self.logger.error(f"❌ Error: {type(error).__name__}: {str(error)}", 
                         extra={'structured_data': log_data}, exc_info=True)

    def log_warning(self, message: str, context: Dict[str, Any] = None):
        """
        Log warning

        Args:
            message: Warning message
            context: Warning context
        """
        self.performance_metrics['warning_count'] += 1

        log_data = {
            'event': 'warning',
            'message': message,
            'timestamp': time.time()
        }

        if context:
            log_data['context'] = context

        self.logger.warning(f"⚠️ {message}", extra={'structured_data': log_data})

    def log_memory_usage(self, device_id: int = None):
        """Log memory usage"""
        if not torch.cuda.is_available():
            return

        if device_id is None:
            device_ids = list(range(torch.cuda.device_count()))
        else:
            device_ids = [device_id]

        memory_stats = {}
        for dev_id in device_ids:
            with torch.cuda.device(dev_id):
                memory_stats[dev_id] = {
                    'allocated_mb': torch.cuda.memory_allocated() / 1024**2,
                    'reserved_mb': torch.cuda.memory_reserved() / 1024**2,
                    'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2,
                    'max_reserved_mb': torch.cuda.max_memory_reserved() / 1024**2
                }

        log_data = {
            'event': 'memory_usage',
            'memory_stats': memory_stats,
            'timestamp': time.time()
        }

        self.logger.debug("💾 Memory usage", extra={'structured_data': log_data})

    @contextmanager
    def log_duration(self, operation_name: str, level: int = logging.INFO):
        """
        Context manager for logging operation duration

        Args:
            operation_name: Operation name
            level: Logging level
        """
        start_time = time.time()
        self.logger.log(level, f"🚀 Start {operation_name}")

        try:
            yield
            duration = time.time() - start_time
            self.logger.log(level, f"✅ Finished {operation_name} (duration: {duration:.2f}s)")

        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"❌ {operation_name} failed (duration: {duration:.2f}s): {e}")
            raise

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        current_time = time.time()
        total_duration = current_time - self.performance_metrics['start_time']

        stats = self.performance_metrics.copy()
        stats['total_duration'] = total_duration

        if stats['throughput_history']:
            stats['avg_throughput'] = sum(stats['throughput_history']) / len(stats['throughput_history'])
            stats['max_throughput'] = max(stats['throughput_history'])
            stats['min_throughput'] = min(stats['throughput_history'])

        return stats


class ColoredFormatter(logging.Formatter):
    """Colored log formatter"""

    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Purple
        'RESET': '\033[0m'        # Reset
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter"""

    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add structured data
        if hasattr(record, 'structured_data'):
            log_entry.update(record.structured_data)

        # Add exception info
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


def setup_multi_gpu_logging(
    log_dir: str = "logs",
    level: int = logging.INFO,
    enable_console: bool = True,
    enable_file: bool = True,
    enable_json: bool = True
) -> MultiGPULogger:
    """
    Convenience function to set up multi-GPU logging

    Args:
        log_dir: Log directory
        level: Logging level
        enable_console: Enable console output
        enable_file: Enable file output
        enable_json: Enable JSON format

    Returns:
        MultiGPULogger: Configured logger
    """
    return MultiGPULogger(
        log_dir=log_dir,
        level=level,
        enable_console=enable_console,
        enable_file=enable_file,
        enable_json=enable_json
    )


def get_training_logger(name: str = "training") -> MultiGPULogger:
    """Get training logger"""
    return MultiGPULogger(
        name=f"unsloth_multigpu.{name}",
        level=logging.INFO,
        enable_console=True,
        enable_file=True,
        enable_json=True
    )


class ProgressTracker:
    """Training progress tracker"""

    def __init__(self, logger: MultiGPULogger, total_steps: int):
        """
        Initialize progress tracker

        Args:
            logger: Logger
            total_steps: Total number of steps
        """
        self.logger = logger
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.log_interval = 10  # Log every 10 steps

    def update(self, step: int, loss: float = None, metrics: Dict = None):
        """
        Update progress

        Args:
            step: Current step
            loss: Loss value
            metrics: Other metrics
        """
        self.current_step = step
        current_time = time.time()

        # Calculate progress
        progress = step / self.total_steps if self.total_steps > 0 else 0
        elapsed_time = current_time - self.start_time

        # Estimate remaining time
        if step > 0:
            avg_time_per_step = elapsed_time / step
            remaining_steps = self.total_steps - step
            eta = remaining_steps * avg_time_per_step
        else:
            eta = 0

        # Should log?
        should_log = (
            step % self.log_interval == 0 or
            step == self.total_steps or
            current_time - self.last_log_time > 30  # At least log every 30 seconds
        )

        if should_log:
            progress_data = {
                'step': step,
                'total_steps': self.total_steps,
                'progress_percent': progress * 100,
                'elapsed_time': elapsed_time,
                'eta': eta,
                'steps_per_second': step / elapsed_time if elapsed_time > 0 else 0
            }

            if loss is not None:
                progress_data['loss'] = loss

            if metrics:
                progress_data['metrics'] = metrics

            self.logger.logger.info(
                f"📊 Progress: {step}/{self.total_steps} ({progress*100:.1f}%) "
                f"ETA: {eta/60:.1f}min",
                extra={'structured_data': {'event': 'progress_update', **progress_data}}
            )

            self.last_log_time = current_time

    def finish(self):
        """Finish training"""
        total_time = time.time() - self.start_time
        self.logger.logger.info(
            f"🎉 Training finished! Total time: {total_time/60:.1f}min, "
            f"Average speed: {self.current_step/total_time:.2f} steps/s"
        ) 