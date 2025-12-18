"""
Minimal Hydra wrapper for load_openpi.py
Just converts Hydra DictConfig to argparse-like namespace and calls original functions.
"""

from omegaconf import DictConfig
from types import SimpleNamespace
import torch

# Import the original functions from load_openpi.py
from load_model.load_openpi import load_openpi_model, load_openpi_model_config


def load_openpi_model_config_from_hydra(model_cfg: DictConfig):
    """
    Convert Hydra config to argparse namespace and call original load_openpi_model_config.
    This is the ONLY function that changes - just argparse to Hydra conversion.
    """
    # Create a namespace that mimics argparse output
    args = SimpleNamespace()
    
    # Map Hydra config to args (same as parse_args() in load_openpi.py)
    args.config_name = model_cfg.get("config_name", "pi0_maniskill")
    args.mode = model_cfg.get("mode", "maniskill")
    args.tasks = model_cfg.get("tasks", "all")
    args.exp_name = model_cfg.get("exp_name", "test")
    args.checkpoint_suffix = model_cfg.get("checkpoint_suffix", None)
    
    # Training hyperparameters
    args.warmup_steps = model_cfg.get("warmup_steps", None)
    args.peak_lr = model_cfg.get("peak_lr", None)
    args.decay_steps = model_cfg.get("decay_steps", None)
    args.decay_lr = model_cfg.get("decay_lr", None)
    args.batch_size = model_cfg.get("batch_size", None)
    args.num_train_steps = model_cfg.get("num_train_steps", None)
    args.save_interval = model_cfg.get("save_interval", None)
    
    # Model architecture params (flattened in config, not nested under 'model' key)
    args.action_horizon = model_cfg.get("action_horizon", None)
    args.action_chunk = model_cfg.get("action_chunk", None)
    args.num_steps = model_cfg.get("num_steps", None)
    args.discrete_state_input = model_cfg.get("discrete_state_input", None)
    args.skip_state_embedding = model_cfg.get("skip_state_embedding", None)
    
    # Checkpoint path
    args.pytorch_weight_path = model_cfg.get("pytorch_weight_path", None)
    args.model_device = model_cfg.get("model_device", "cuda:0")
    
    # Call the ORIGINAL function from load_openpi.py
    return load_openpi_model_config(args)


def load_openpi_model_from_hydra(model_cfg: DictConfig, model_device: torch.device):
    """
    Load OpenPI model from Hydra config by calling original functions.
    """
    # Get OpenPI config using original logic
    openpi_config = load_openpi_model_config_from_hydra(model_cfg)
    print(f"Loaded openpi_config: {openpi_config}")

    # Call ORIGINAL load_openpi_model function - no changes at all
    model = load_openpi_model(openpi_config, model_device)
    # Save the model architecture to a file
    
    return model

