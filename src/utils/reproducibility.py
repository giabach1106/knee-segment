"""Utilities for ensuring reproducible results."""

import random
import os
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Sets seeds for:
    - Python random
    - NumPy
    - PyTorch (CPU and CUDA)
    
    Args:
        seed: Random seed value.
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Environment variable for some libraries
    os.environ['PYTHONHASHSEED'] = str(seed)


def ensure_reproducibility(
    seed: int = 42,
    deterministic: bool = True,
    benchmark: bool = False
) -> None:
    """Ensure reproducible results with additional PyTorch settings.
    
    Args:
        seed: Random seed value.
        deterministic: Whether to use deterministic algorithms.
        benchmark: Whether to use CUDNN benchmark mode (can improve
            performance but reduces reproducibility).
    """
    # Set seeds
    set_seed(seed)
    
    # PyTorch reproducibility settings
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # For some operations, we need to set this
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True)
            except:
                # Some operations might not have deterministic implementations
                pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = benchmark


def worker_init_fn(worker_id: int) -> None:
    """Initialize worker with unique random seed.
    
    This function should be passed to DataLoader as worker_init_fn
    to ensure that data augmentation is different across workers
    but reproducible across runs.
    
    Args:
        worker_id: Worker ID from DataLoader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed) 