"""Logging utilities for consistent logging across the project."""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
from datetime import datetime


def setup_logging(
    log_dir: Optional[Union[str, Path]] = None,
    log_level: Union[str, int] = logging.INFO,
    console: bool = True,
    filename: Optional[str] = None
) -> logging.Logger:
    """Set up logging configuration for the project.
    
    Args:
        log_dir: Directory to save log files. If None, only console logging.
        log_level: Logging level (e.g., logging.INFO, 'DEBUG').
        console: Whether to log to console.
        filename: Custom log filename. If None, uses timestamp.
        
    Returns:
        Root logger instance.
    """
    # Convert string log level to int if needed
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper())
        
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
    # File handler
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"knee_segmentation_{timestamp}.log"
            
        log_file = log_dir / filename
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        root_logger.info(f"Logging to file: {log_file}")
        
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name.
    
    Args:
        name: Logger name (typically __name__).
        
    Returns:
        Logger instance.
    """
    return logging.getLogger(name) 