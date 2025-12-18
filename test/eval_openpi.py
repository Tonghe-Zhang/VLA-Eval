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

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import logging
from quick_test_maniskill.test.eval_base import BaseEnvTester
from quick_test_maniskill.env.fetch_rgb_from_obs import fetch_rgb_from_obs_allenvs 

# Check OpenPI availability
try:
    import openpi
    OPENPI_AVAILABLE = True
except ImportError:
    OPENPI_AVAILABLE = False
    logging.warning("Warning: Could not import OpenPI.")

# Recombine batch
def stack_samples(*samples):
    return torch.from_numpy(np.stack(samples))
    

# Convert to torch and move to device (recursively for nested dicts)
def to_device(x, model_device):
    if isinstance(x, dict):
        return {k: to_device(v, model_device) for k, v in x.items()}
    elif isinstance(x, torch.Tensor):
        return x.to(model_device)
    else:
        return x

class Pi0EnvTester(BaseEnvTester):
    """
    Tester that uses a Pi0/Pi0.5 model for action generation.
    Uses Hydra configs for both environment and model configuration.
    """
    def __init__(self, cfg: DictConfig):
        if not OPENPI_AVAILABLE:
            raise ImportError("OpenPI modules are not available.")
        
        # Extract model_cfg BEFORE passing to base class
        if "model_cfg" not in cfg:
            raise ValueError(f"No 'model_cfg' in config. Available keys: {list(cfg.keys())}")
        
        self.model_cfg = cfg.model_cfg
        
        # Remove model_cfg before passing to base class
        env_cfg = OmegaConf.create({k: v for k, v in cfg.items() if k != 'model_cfg'})
        super().__init__(env_cfg)
        
        # Load model using model config
        self.load_model()

    def load_model(self):
        """Load model and transforms - following train_pytorch.py approach."""
        from load_model.load_openpi_from_hydra import load_openpi_model_from_hydra
        from openpi.training import config as train_config
        from openpi.shared import normalize
        import openpi.transforms as transforms
        
        logging.info(f"Model config:\n{OmegaConf.to_yaml(self.model_cfg)}")
        
        model_device_str = self.model_cfg.get("model_device", "cuda:0")
        self.model_device = torch.device(model_device_str)
        
        # Load model
        logging.info("Loading model...")
        self.model = load_openpi_model_from_hydra(self.model_cfg, self.model_device)
        # Save the model architecture to a file

        self.model.eval()
        
        # Get train config (same as train_pytorch.py does)
        config_name = self.model_cfg.get("config_name", "pi05_maniskill")
        train_cfg = train_config.get_config(config_name)
        
        # Create data config (same as build_datasets in train_pytorch.py)
        data_cfg = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
        
        # Override norm_stats if specified from command line
        if "norm_stats_dir" in self.model_cfg and self.model_cfg.norm_stats_dir is not None:
            norm_stats = normalize.load(self.model_cfg.norm_stats_dir)
            logging.info(f"✓ Loaded norm stats from: {self.model_cfg.norm_stats_dir}: \n {norm_stats}")
        else:
            norm_stats = data_cfg.norm_stats
            logging.info(f"✓ Loaded norm stats from config:\n {norm_stats}")
        
        # Build transforms exactly as training does
        # Input: data_transforms.inputs -> Normalize -> model_transforms.inputs
        self.input_transform = transforms.compose([
            *data_cfg.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_cfg.use_quantile_norm),
            *data_cfg.model_transforms.inputs,
        ])
        
        # Output: Unnormalize -> data_transforms.outputs
        self.output_transform = transforms.compose([
            transforms.Unnormalize(norm_stats, use_quantiles=data_cfg.use_quantile_norm),
            *data_cfg.data_transforms.outputs,
        ])
        
        logging.info("✓ Model and transforms loaded")
        
    def get_language_instruction(self, n_envs: int) -> list[str]:
        # TODO: Human real-time input from voice or keyboard
        # Get instructions from environment if available, or custom instructions
        if hasattr(self.env.unwrapped, "get_language_instruction"):
            return self.env.unwrapped.get_language_instruction()
        return [f"Do something useful with the object in the scene, robot arm and grippers. " for _ in range(n_envs)]

    def get_action(self, obs: dict, proprioception: torch.Tensor, language_instruction: list[str]) -> torch.Tensor:
        """Get action with transforms."""
        import openpi.models.model as _model
        
        # Get RGB image
        rgb_dict = fetch_rgb_from_obs_allenvs(self.env_type, obs, self.sim_device, self.model_device, normalize=True)
        image = next(iter(rgb_dict.values())) if isinstance(rgb_dict, dict) else rgb_dict
        
        # Convert tensors to CPU numpy for transforms
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if isinstance(proprioception, torch.Tensor):
            proprioception = proprioception.cpu().numpy()
        
        # Get batch size (could be num_envs or temporal window)
        batch_size = image.shape[0] if image.ndim == 4 else 1
        
        # Process each sample through transform pipeline
        transformed_samples = []
        for i in range(batch_size):
            # Extract single sample
            sample_image = image[i] if batch_size > 1 else image
            sample_state = proprioception[i] if batch_size > 1 and proprioception.ndim > 1 else proprioception
            
            # Transpose image from [C, H, W] to [H, W, C]
            if sample_image.ndim == 3 and sample_image.shape[0] == 3:
                sample_image = sample_image.transpose(1, 2, 0)
            
            # Create obs dict for this sample
            sample_obs = {
                "observation/image": sample_image,
                "observation/state": sample_state,
                "prompt": language_instruction[0] if isinstance(language_instruction, list) else language_instruction,
            }
            
            # Apply transforms to single sample
            transformed = self.input_transform(sample_obs)
            transformed_samples.append(transformed)
        

        
        import jax
        processed = jax.tree.map(stack_samples, *transformed_samples)
        
        
        processed = to_device(processed, self.model_device)
        
        # Model inference
        observation = _model.Observation.from_dict(processed)
        with torch.no_grad():
            actions = self.model.sample_actions(
                self.model_device,
                observation=observation,
                noise=None,
                num_steps=self.model.config.num_steps
            )
        
        # Apply output transforms (Unnormalize -> SimplerOutputs)
        output = self.output_transform({
            "actions": actions.cpu().numpy(),
            "state": processed["state"].cpu().numpy(),
        })

        actions = torch.from_numpy(output["actions"]).to(self.sim_device) 
        actions = actions[:,:,:self.single_action_dim]
        return actions

@hydra.main(version_base=None, config_path=".", config_name=None)
def main(cfg: DictConfig):
    tester = Pi0EnvTester(cfg)        
    tester.run()

if __name__ == "__main__":
    main()
