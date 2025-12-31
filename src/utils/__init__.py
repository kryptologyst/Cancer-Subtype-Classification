"""Utility functions for cancer subtype classification project."""

import os
import random
import logging
from typing import Any, Dict, Optional, Union
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def setup_logging(level: str = "INFO", log_dir: str = "logs") -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory to save log files
        
    Returns:
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'cancer_classification.log')),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized")
    return logger


def set_deterministic_seed(seed: int = 42) -> None:
    """Set deterministic seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # For MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps'):
        torch.mps.manual_seed(seed)


def get_device(device: str = "auto") -> torch.device:
    """Get the appropriate device for computation.
    
    Args:
        device: Device preference ("auto", "cpu", "cuda", "mps")
        
    Returns:
        PyTorch device object
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    return torch.device(device)


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        OmegaConf configuration object
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config = OmegaConf.load(config_path)
    return config


def save_config(config: DictConfig, save_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration object to save
        save_path: Path to save configuration file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    OmegaConf.save(config, save_path)


def create_directories(directories: list[str]) -> None:
    """Create directories if they don't exist.
    
    Args:
        directories: List of directory paths to create
    """
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def get_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Calculate class weights for imbalanced datasets.
    
    Args:
        y: Target labels
        
    Returns:
        Dictionary mapping class indices to weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    unique_classes = np.unique(y)
    class_weights = compute_class_weight(
        'balanced',
        classes=unique_classes,
        y=y
    )
    
    return dict(zip(unique_classes, class_weights))


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score: float, model: Any) -> bool:
        """Check if training should stop.
        
        Args:
            score: Current validation score
            model: Model to potentially save weights from
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = model.state_dict() if hasattr(model, 'state_dict') else None
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict() if hasattr(model, 'state_dict') else None
        
        return False
