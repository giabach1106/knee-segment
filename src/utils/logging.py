"""Utilities logging cho project."""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(log_dir=None, log_level=logging.INFO, console=True, filename=None):
    """Setup logging"""
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper())
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers = []
    
    # Console output
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File output 
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        if not filename:
            timestamp = datetime.now().strftime("%m%d_%H%M")
            filename = f"knee_seg_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_dir / filename)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger


def get_logger(name):
    """Get logger instance."""
    return logging.getLogger(name) 