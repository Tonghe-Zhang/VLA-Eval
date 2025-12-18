import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import torch
from pathlib import Path
from datetime import datetime
from PIL import Image
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

from ManiSkill.mani_skill.envs.sapien_env import BaseEnv  
from ManiSkill.mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from functools import partial
import sys
script_dir = Path(__file__).parent  # test_other_things/
test_dir = script_dir.parent  # test/
quick_test_dir = test_dir.parent  # quick_test_maniskill/
sys.path.insert(0, str(quick_test_dir))

from quick_test_maniskill.env.env_helpers import setup_maniskill_env
from quick_test_maniskill.env.multi_action_wrapper import MultiActionWrapper
from quick_test_maniskill.env.per_step_reward_wrapper import PerStepRewardWrapper
from quick_test_maniskill.env.fetch_rgb_from_obs import fetch_rgb_from_obs_allenvs
from quick_test_maniskill.evaluate.eval_helpers_versatile import (
    create_batch_episode_data, 
    create_batch_videos,
    tile_images
)


@hydra.main(version_base=None, config_path="env_cfg", config_name="stack_cubes")
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
    
    action_chunk_size = cfg.action_chunk_size
    single_action_dim = cfg.single_action_dim
    
    # Setup environment wrappers
    wrappers = [
        partial(PerStepRewardWrapper),
        partial(MultiActionWrapper),
    ]
    
    # Create environment
    env: ManiSkillVectorEnv = setup_maniskill_env(
        env_id, 
        n_envs, 
        n_steps_episode, 
        sim_backend, 
        sim_device, 
        sim_device_id, 
        sim_config, 
        sensor_config, 
        obs_mode, 
        control_mode, 
        episode_mode, 
        wrappers=wrappers
    )
    env_unwrapp: BaseEnv = env.unwrapped
    
    # Create output directories (relative to script location)
    timestamp = datetime.now().strftime("%H-%M-%S_%d-%m-%Y")
    base_dir = script_dir.parent / "log" / "quick_test_maniskill"
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
    print(f"obs={obs.keys()}, info={info.keys()}")  # type: ignore
    
    # Save initial observation images
    rgb_obs = fetch_rgb_from_obs_allenvs(env_type, obs, sim_device, model_device)  # Dict[str, torch.Tensor]
    print(f"RGB observation keys: {list(rgb_obs.keys())}")
    for camera_name, img in rgb_obs.items():
        print(f"  Camera: {camera_name}, Image shape: {img.shape}")
    
    # Save tiled image of all initial observations
    for camera_name, img in rgb_obs.items():
        tiled_init_obs = tile_images(img.permute(0, 2, 3, 1).cpu().numpy())
        init_obs_path = init_obs_dir / f"{camera_name}.png"
        Image.fromarray(tiled_init_obs).save(init_obs_path)
        print(f"Saved initial observation image to: {init_obs_path}")
    
    # Run test episodes with video recording
    print(f"\nRunning {num_test_steps} test steps with video recording...")
    
    # Create dummy instructions for each environment
    instructions = [f"Test env {i}" for i in range(n_envs)]
    
    # Create batch episode data recorder
    batch_data = dict()
    video_dir_camera_dir = dict()
    for camera_name in rgb_obs.keys():
        batch_data[camera_name] = create_batch_episode_data(num_envs=n_envs, instructions=instructions)
        video_dir_camera_dir[camera_name] = video_test_dir / f"{camera_name}"
        video_dir_camera_dir[camera_name].mkdir(parents=True, exist_ok=True)
    
    # Run episode loop
    for step in tqdm(range(num_test_steps)):
        # Generate random actions for testing
        actions = torch.randn(n_envs, action_chunk_size, single_action_dim, device=sim_device)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(actions)
        
        # Record step data
        obs_rgb = fetch_rgb_from_obs_allenvs(env_type, obs, sim_device, model_device)
        
        for camera_name in obs_rgb.keys():
            batch_data[camera_name].add_step_data_in_batch(
                obs_rgb_batch=obs_rgb[camera_name].permute(0, 2, 3, 1),
                rewards=reward,
                step_info=info,
                actions_batch=actions
            )
        
        # Check if any environment is done
        if terminated.any() or truncated.any():
            print(f"  Some environments terminated at step {step}")
    
    # Add final observation
    for camera_name in batch_data.keys():
        final_obs_rgb = fetch_rgb_from_obs_allenvs(env_type, obs, sim_device, model_device)[camera_name]
        batch_data[camera_name].add_final_observation_in_batch(final_obs_rgb.permute(0, 2, 3, 1))
    
    # Create and save video
    print(f"\nCreating video...")
    for camera_name in batch_data.keys():
        create_batch_videos(batch_data[camera_name], video_dir_camera_dir[camera_name], fps=frame_per_second, filter_success=False)
    
    print(f"\n{'='*60}")
    print(f"Test completed successfully!")
    print(f"  - Initial observation saved to: {init_obs_path}")
    print(f"  - Video saved to: {video_test_dir}")
    print(f"  - Total steps: {num_test_steps}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
