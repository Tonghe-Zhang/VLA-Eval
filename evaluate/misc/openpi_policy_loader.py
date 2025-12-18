"""Minimal wrapper to load OpenPI policy with proper transforms."""
import pathlib
import torch
from openpi.policies import policy_config
from openpi.training import config as train_config


def load_policy_from_checkpoint(config_name: str, checkpoint_dir: str, device: str = "cuda:0"):
    """Load OpenPI policy with all transforms from checkpoint.
    
    Args:
        config_name: Training config name (e.g., "pi05_maniskill")
        checkpoint_dir: Path to checkpoint directory
        device: Device string
        
    Returns:
        OpenPI Policy object with input/output transforms
    """
    cfg = train_config.get_config(config_name)
    policy = policy_config.create_trained_policy(
        train_config=cfg,
        checkpoint_dir=pathlib.Path(checkpoint_dir),
        pytorch_device=device,
    )
    return policy

