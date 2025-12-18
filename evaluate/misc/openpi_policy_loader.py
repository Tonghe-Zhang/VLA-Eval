# MIT License

# Copyright (c) 2025 Tonghe Zhang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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

