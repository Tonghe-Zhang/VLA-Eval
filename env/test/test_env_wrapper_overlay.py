import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import torch
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import shutil

from setup_maniskill_env import setup_maniskill_env
from ManiSkill.mani_skill.envs.sapien_env import BaseEnv  
from ManiSkill.mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

# Register custom SimplerEnvPlus environments
import sys
project_root = Path(__file__).parent.parent.parent.parent  # Navigate to Project/
maniskill_root = project_root / "Manip" / "ManiSkill"
sys.path.insert(0, str(maniskill_root))

# Import custom environment definitions to register them with ManiSkill
# This will trigger the @register_env decorators in each module
try:
    # Import base environments
    import SimplerEnvPlus.tasks.put_carrot_on_plate
    import SimplerEnvPlus.tasks.put_on_in_scene_multi
    # Import all variant environments
    import SimplerEnvPlus.tasks.variants
    print("Successfully registered custom SimplerEnvPlus environments")
except ImportError as e:
    print(f"Warning: Could not import SimplerEnvPlus tasks: {e}")
    print("Custom environments may not be available")

from functools import partial
from env.multi_action_wrapper import MultiActionWrapper
from env.per_step_reward_wrapper import PerStepRewardWrapper

from env.test.fetch_rgb_from_obs import fetch_rgb_from_obs_allenvs
import sys
script_dir = Path(__file__).parent  # test/
env_dir = script_dir.parent  # env/
quick_test_dir = env_dir.parent  # quick_test_maniskill/
sys.path.insert(0, str(quick_test_dir))

from evaluate.eval_helpers import (
    create_batch_episode_data, 
    create_batch_videos,
    tile_images
)


@hydra.main(version_base=None, config_path="env_configs", config_name="stack_cubes")
def main(cfg: DictConfig):
    """Main function using Hydra configuration management."""
    
    # Print configuration
    print("=" * 60)
    print("Configuration loaded:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)
    
    # Extract configuration values
    env_id = cfg.env_id
    env_type = cfg.env_type
    control_mode = cfg.control_mode
    episode_mode = cfg.episode_mode
    
    sim_config = OmegaConf.to_container(cfg.sim_config, resolve=True)
    sensor_config = OmegaConf.to_container(cfg.sensor_config, resolve=True)
    
    obs_mode = cfg.obs_mode
    n_envs = cfg.n_envs
    n_steps_episode = cfg.n_steps_episode
    num_test_steps = cfg.num_test_steps
    frame_per_second = cfg.frame_per_second
    
    sim_backend = cfg.sim_backend
    sim_device = cfg.sim_device
    sim_device_id = cfg.sim_device_id
    model_device = cfg.model_device
    render_mode = cfg.render_mode
    action_chunk_size = cfg.action_chunk_size
    single_action_dim = cfg.single_action_dim
    obj_set = cfg.get("obj_set", None)

    # Setup environment wrappers
    wrappers = [
        partial(PerStepRewardWrapper),
        partial(MultiActionWrapper),
    ]
    
    # Create environment
    env: ManiSkillVectorEnv = setup_maniskill_env(
        env_id=env_id, 
        n_envs=n_envs, 
        max_episode_len=n_steps_episode, 
        sim_backend=sim_backend, 
        sim_device=sim_device, 
        sim_device_id=sim_device_id, 
        sim_config=sim_config, 
        sensor_config=sensor_config, 
        obs_mode=obs_mode, 
        render_mode=render_mode,
        control_mode=control_mode, 
        episode_mode=episode_mode, 
        obj_set=obj_set,
        wrappers=wrappers,
    )
    env_unwrapp: BaseEnv = env.unwrapped
    
    # Create output directories (relative to script location)
    timestamp = datetime.now().strftime("%H-%M-%S_%d-%m-%Y")
    base_dir = script_dir.parent.parent / "log" / "quick_test_maniskill"
    init_obs_dir = base_dir / "figs" / "init-obs" / f"{env_id}" / f"{timestamp}"
    video_test_dir = base_dir / "video_test" / f"{env_id}" / f"{timestamp}"
    init_obs_dir.mkdir(parents=True, exist_ok=True)
    video_test_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directories created:")
    print(f"  Init obs dir: {init_obs_dir}")
    print(f"  Video dir: {video_test_dir}")
    
    # Test env.reset() and save initial observations
    print(f"\nTest env.reset()")
    obs, info = env.reset()
    reward = torch.zeros(n_envs, device=sim_device)
    print(f"obs={obs.keys()}, info={info.keys()}")  # type: ignore


    # Save initial observation images
    obs_rgb_overlay = fetch_rgb_from_obs_allenvs(env_type, obs, sim_device, model_device, info=info, reward=reward, normalize=False)
    print(f"RGB observation keys: {list(obs_rgb_overlay.keys())}")
    for camera_name, img in obs_rgb_overlay.items():
        print(f"  Camera: {camera_name}, Image shape: {img.shape}")
    
    # Save tiled image of all initial observations
    for camera_name, img in obs_rgb_overlay.items():
        img_tiled = img.permute(0, 2, 3, 1).cpu().numpy() # [N,H,W,C] -> [N,C,H,W]
        tiled_init_obs = tile_images(img_tiled)
        init_obs_path = init_obs_dir / f"{camera_name}.png"
        Image.fromarray(tiled_init_obs).save(init_obs_path)
        print(f"Saved initial observation image to: {init_obs_path}")
    
    # Copy Hydra config to init obs directory immediately after saving
    hydra_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    hydra_config_src = hydra_output_dir / ".hydra"
    
    if hydra_config_src.exists():
        hydra_dest_init = init_obs_dir / ".hydra"
        shutil.copytree(hydra_config_src, hydra_dest_init, dirs_exist_ok=True)
        print(f"✓ Copied Hydra config to init obs directory: {hydra_dest_init}")

    # Run test episodes with video recording
    print(f"\nRunning {num_test_steps} test steps with video recording...")
    
    # Create dummy instructions for each environment
    instructions = [f"Test env {i}" for i in range(n_envs)]
    
    # Create batch episode data recorder
    batch_data = dict()
    video_dir_camera_dir = dict()
    for camera_name in obs_rgb_overlay.keys():
        batch_data[camera_name] = create_batch_episode_data(num_envs=n_envs, instructions=instructions)
        video_dir_camera_dir[camera_name] = video_test_dir / f"{camera_name}"
        video_dir_camera_dir[camera_name].mkdir(parents=True, exist_ok=True)
    
    # Run episode loop
    for step in tqdm(range(num_test_steps)):
        # Generate random actions for testing
        actions = torch.randn(n_envs, action_chunk_size, single_action_dim, device=sim_device)
        
        obs, reward, terminated, truncated, info = env.step(actions)
        
        obs_rgb_overlay = fetch_rgb_from_obs_allenvs(env_type, obs, sim_device, model_device, info=info, reward=reward, normalize=False)
        

        for camera_name in obs_rgb_overlay.keys():
            batch_data[camera_name].add_step_data_in_batch(
                obs_rgb_batch=obs_rgb_overlay[camera_name].permute(0, 2, 3, 1),
                rewards=reward,
                step_info=info,
                actions_batch=actions
            )
        
        # Check if any environment is done
        if terminated.any() or truncated.any():
            print(f"  Some environments terminated at step {step}")
    
    # Create and save video
    print(f"\nCreating video...")
    for camera_name in batch_data.keys():
        custom_filename= f"{n_envs}envs"
        create_batch_videos(batch_data[camera_name], video_dir_camera_dir[camera_name], fps=frame_per_second, filter_success=False, custom_filename=custom_filename)
    
    # Copy Hydra config to video directory immediately after creating videos
    hydra_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    hydra_config_src = hydra_output_dir / ".hydra"
    
    if hydra_config_src.exists():
        hydra_dest_video = video_test_dir / ".hydra"
        shutil.copytree(hydra_config_src, hydra_dest_video, dirs_exist_ok=True)
        print(f"✓ Copied Hydra config to video directory: {hydra_dest_video}")
    
    print(f"\n{'='*60}")
    print(f"Test completed successfully!")
    print(f"  - Initial observation saved to: {init_obs_dir}")
    print(f"  - Video saved to: {video_test_dir}")
    print(f"  - Configs saved to both directories")
    print(f"  - Total steps: {num_test_steps}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
