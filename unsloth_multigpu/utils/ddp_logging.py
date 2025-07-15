"""
DDP Logging utilities
Provides rank-aware logging functions for distributed training
"""

import logging
import os
from functools import lru_cache
from typing import Any


def get_rank() -> int:
    """Get current process rank"""
    return int(os.getenv("RANK", "0"))


def get_local_rank() -> int:
    """Get current local rank"""
    return int(os.getenv("LOCAL_RANK", "0"))


def is_main_process() -> bool:
    """Check if current process is the main process (rank 0)"""
    return get_rank() == 0


def info_rank0(logger: logging.Logger, *args, **kwargs) -> None:
    """Log info message only from rank 0 process"""
    if is_main_process():
        logger.info(*args, **kwargs)


def warning_rank0(logger: logging.Logger, *args, **kwargs) -> None:
    """Log warning message only from rank 0 process"""
    if is_main_process():
        logger.warning(*args, **kwargs)


def error_rank0(logger: logging.Logger, *args, **kwargs) -> None:
    """Log error message only from rank 0 process"""
    if is_main_process():
        logger.error(*args, **kwargs)


def debug_rank0(logger: logging.Logger, *args, **kwargs) -> None:
    """Log debug message only from rank 0 process"""
    if is_main_process():
        logger.debug(*args, **kwargs)


@lru_cache(None)
def warning_rank0_once(logger: logging.Logger, *args, **kwargs) -> None:
    """Log warning message only once from rank 0 process"""
    if is_main_process():
        logger.warning(*args, **kwargs)


def info_all_ranks(logger: logging.Logger, message: str, *args, **kwargs) -> None:
    """Log info message from all ranks with rank prefix"""
    rank = get_rank()
    local_rank = get_local_rank()
    logger.info(f"[Rank {rank}/Local {local_rank}] {message}", *args, **kwargs)


# Monkey patch logging.Logger to add rank-aware methods
def _patch_logger():
    """Add rank-aware methods to logging.Logger class"""
    
    def info_rank0_method(self, *args, **kwargs):
        info_rank0(self, *args, **kwargs)
    
    def warning_rank0_method(self, *args, **kwargs):
        warning_rank0(self, *args, **kwargs)
    
    def error_rank0_method(self, *args, **kwargs):
        error_rank0(self, *args, **kwargs)
    
    def debug_rank0_method(self, *args, **kwargs):
        debug_rank0(self, *args, **kwargs)
    
    def warning_rank0_once_method(self, *args, **kwargs):
        warning_rank0_once(self, *args, **kwargs)
    
    def info_all_ranks_method(self, message, *args, **kwargs):
        info_all_ranks(self, message, *args, **kwargs)
    
    # Add methods to Logger class
    logging.Logger.info_rank0 = info_rank0_method
    logging.Logger.warning_rank0 = warning_rank0_method
    logging.Logger.error_rank0 = error_rank0_method
    logging.Logger.debug_rank0 = debug_rank0_method
    logging.Logger.warning_rank0_once = warning_rank0_once_method
    logging.Logger.info_all_ranks = info_all_ranks_method


# Apply the patch
_patch_logger()


class DDPLogger:
    """
    DDP-aware logger wrapper
    
    Usage:
        from unsloth_multigpu.utils.ddp_logging import DDPLogger
        
        logger = DDPLogger(__name__)
        logger.info_rank0("This only shows on rank 0")
        logger.info_all_ranks("This shows on all ranks with rank prefix")
    """
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.is_main = is_main_process()
    
    def info_rank0(self, *args, **kwargs):
        """Log info message only from rank 0"""
        if self.is_main:
            self.logger.info(*args, **kwargs)
    
    def warning_rank0(self, *args, **kwargs):
        """Log warning message only from rank 0"""
        if self.is_main:
            self.logger.warning(*args, **kwargs)
    
    def error_rank0(self, *args, **kwargs):
        """Log error message only from rank 0"""
        if self.is_main:
            self.logger.error(*args, **kwargs)
    
    def debug_rank0(self, *args, **kwargs):
        """Log debug message only from rank 0"""
        if self.is_main:
            self.logger.debug(*args, **kwargs)
    
    def info_all_ranks(self, message: str, *args, **kwargs):
        """Log info message from all ranks with rank prefix"""
        self.logger.info(f"[Rank {self.rank}/Local {self.local_rank}] {message}", *args, **kwargs)
    
    def warning_all_ranks(self, message: str, *args, **kwargs):
        """Log warning message from all ranks with rank prefix"""
        self.logger.warning(f"[Rank {self.rank}/Local {self.local_rank}] {message}", *args, **kwargs)
    
    # Forward other methods to the original logger
    def __getattr__(self, name: str) -> Any:
        return getattr(self.logger, name)


def get_ddp_logger(name: str) -> DDPLogger:
    """Get a DDP-aware logger"""
    return DDPLogger(name)