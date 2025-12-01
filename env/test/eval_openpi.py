
import hydra
from omegaconf import DictConfig
import argparse
import torch
from quick_test_maniskill.evaluate.load_openpi import load_openpi_model_config, load_openpi_model
from quick_test_maniskill.env.test.eval_base import BaseEnvTester
from quick_test_maniskill.env.test.fetch_rgb_from_obs import fetch_rgb_from_obs_allenvs
from ManiSkill.mani_skill.envs.sapien_env import BaseEnv
from ManiSkill.mani_skill.utils.structs.articulation import Articulation 

# Importing OpenPI modules
try:
    from evaluate.load_openpi import load_openpi_model, load_openpi_model_config
    OPENPI_AVAILABLE = True
    print("Successfully imported OpenPI modules.")
except ImportError:
    print("Warning: Could not import OpenPI modules. Pi0EnvTester will not be available.")
    OPENPI_AVAILABLE = False

class Pi0EnvTester(BaseEnvTester):
    """
    Tester that uses a Pi0 model for action generation.
    """
    def __init__(self, cfg: DictConfig):
        if not OPENPI_AVAILABLE:
            raise ImportError("OpenPI modules are not available.")
        super().__init__(cfg)
        self.load_model()

    def load_model(self):
        """Load the Pi0 model using OpenPI loader."""
        # Construct args for load_openpi_model_config
        # Note: We assume relevant config keys are present in self.cfg or passed properly
        # For this example, we'll try to read from cfg or set defaults
        
        # Create a namespace to mimic argparse args
        args = argparse.Namespace()
        
        # Map cfg values to args or defaults
        # We expect cfg to possibly contain 'model' section or we use defaults
        args.config_name = self.cfg.get("model_config_name", "pi0_maniskill") # Default config name
        args.mode = self.cfg.get("mode", "few")
        args.tasks = self.cfg.get("tasks", "all")
        args.exp_name = self.cfg.get("exp_name", "test")
        args.checkpoint_suffix = self.cfg.get("checkpoint_suffix", None)
        
        # Explicit overrides from cfg
        args.warmup_steps = None
        args.peak_lr = None
        args.decay_steps = None
        args.decay_lr = None
        args.batch_size = None
        args.discrete_state_input = None
        args.action_horizon = None
        args.action_chunk = Noneskip_state_embedding
        
        # Important: weights path
        # Check if cfg has a model path
        args.pytorch_weight_path = self.cfg.get("model_path", None)
        args.model_device = self.cfg.get("model_device", "cuda:0")
        
        print("Loading OpenPI model config...")
        self.model_config = load_openpi_model_config(args)
        
        print("Loading Pi0 model...")
        self.model = load_openpi_model(self.model_config, torch.device(args.model_device))
        self.model.eval()
        
    def get_language_instruction(self, n_envs: int) -> list[str]:
        # TODO: Human real-time input from voice or keyboard
        # Get instructions from environment if available, or custom instructions
        if hasattr(self.env.unwrapped, "get_language_instruction"):
            return self.env.unwrapped.get_language_instruction()
        return [f"Do something useful with the object in the scene and the robot arm and gripper. "]

    def get_action(self, obs, info) -> torch.Tensor:
        # Prepare batch for model
        rgb_image = fetch_rgb_from_obs_allenvs(
            self.env_type, obs, self.sim_device, self.model_device, normalize=True
        )
        # Assume single camera for now or pick '3rd_view_camera' / 'base_camera'
        # Pi0 usually expects "observation.images.top"
        
        # Identify the main camera
        cam_key = "3rd_view_camera" if "3rd_view_camera" in rgb_image else "base_camera"
        if cam_key not in rgb_image and len(rgb_image) > 0:
            cam_key = list(rgb_image.keys())[0]
            
        img_tensor = rgb_image[cam_key] # [B, C, H, W]
        
        proprioception = self.get_proprioception(obs, self.env)
        instructions = self.get_language_instruction(self.n_envs)
        
        batch = {
            "observation.images.top": img_tensor,
            "observation.state": proprioception,
            "task": instructions,
        }
        
        with torch.no_grad():
            action = self.model.select_action(batch)
            
        # Ensure action is on sim device
        return action.to(self.sim_device)

@hydra.main(version_base=None, config_path="env_configs", config_name="stack_cubes")
def main(cfg: DictConfig):
    tester = Pi0EnvTester(cfg)        
    tester.run()

if __name__ == "__main__":
    main()
