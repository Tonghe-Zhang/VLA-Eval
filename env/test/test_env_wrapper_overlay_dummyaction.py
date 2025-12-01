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
import shutil

from setup_maniskill_env import setup_maniskill_env
from ManiSkill.mani_skill.envs.sapien_env import BaseEnv  
from ManiSkill.mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from ManiSkill.mani_skill.utils import common  # NEW: for to_numpy

# Register custom SimplerEnvPlus environments
import sys
project_root = Path(__file__).parent.parent.parent.parent  # Navigate to Project/
maniskill_root = project_root / "Manip" / "ManiSkill"
sys.path.insert(0, str(maniskill_root))

# Import custom environment definitions to register them with ManiSkill to trigger the @register_env decorators in each module
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


def get_rgb_obs_with_info_overlay(env, obs, info, reward, env_type, sim_device, model_device, use_render_camera=False):
    """
    Get RGB observations with info overlay, either from render() or sensor_data.
    
    Args:
        env: The environment
        obs: Observation dict
        info: Info dict
        reward: Reward tensor
        env_type: Type of environment
        sim_device: Simulation device
        model_device: Model device
        use_render_camera: If True, use render() camera; if False, use observation camera
        
    Returns:
        Dict mapping camera names to RGB tensors with overlays [B, C, H, W]
    """
    if use_render_camera:
        # Use render() camera - better for visualization
        rendered_img = env.render()  # Returns numpy array (B, H, W, C) or (H, W, C)
        rendered_img = common.to_numpy(rendered_img)
        if len(rendered_img.shape) == 3:
            rendered_img = rendered_img[None]  # Add batch dim if single env
        
        # Convert to torch and permute to (B, C, H, W)
        rendered_img_torch = torch.from_numpy(rendered_img).to(sim_device)
        rendered_img_torch = rendered_img_torch.permute(0, 3, 1, 2)
        
        # Create camera dict
        obs_rgb_overlay = {"render_camera": rendered_img_torch}
        
        # Add overlays
        from env.test.fetch_rgb_from_obs import overlay_info_on_rgb_image
        obs_rgb_overlay["render_camera"] = overlay_info_on_rgb_image(
            obs_rgb_overlay["render_camera"], info, reward
        )
    else:
        # Use observation camera - what agent sees
        obs_rgb_overlay = fetch_rgb_from_obs_allenvs(
            env_type, obs, sim_device, model_device, info=info, reward=reward, normalize=False
        )
    
    return obs_rgb_overlay

def flatten_config_if_nested(cfg: DictConfig) -> DictConfig:
    """Flatten Hydra configs that are wrapped inside group names."""
    if len(cfg) == 1 and not any(k in cfg for k in ['env_id', 'env_type']):
        group_name = list(cfg.keys())[0]
        print(f"Flattening nested config group: {group_name}")
        return cfg[group_name]
    return cfg

def prepare_output_dirs(log_root: Path, env_id: str):
    """Create timestamped directories for initial observations and videos."""
    timestamp = datetime.now().strftime("%Y-%d-%m--%H-%M-%S")
    init_obs_dir = log_root / "figs" / "init-obs" / f"{env_id}" / f"{timestamp}"
    video_test_dir = log_root / "video_test" / f"{env_id}" / f"{timestamp}"
    init_obs_dir.mkdir(parents=True, exist_ok=True)
    video_test_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directories created:")
    print(f"  Init obs dir: {init_obs_dir}")
    print(f"  Video dir: {video_test_dir}")
    return timestamp, init_obs_dir, video_test_dir

def save_initial_observations_to_disk(obs_rgb_overlay, init_obs_dir: Path):
    """Persist tiled initial RGB observations per camera."""
    for camera_name, img in obs_rgb_overlay.items():
        img_tiled = img.permute(0, 2, 3, 1).cpu().numpy()
        tiled_init_obs = tile_images(img_tiled)
        init_obs_path = init_obs_dir / f"{camera_name}.png"
        Image.fromarray(tiled_init_obs).save(init_obs_path)
        print(f"Saved initial observation image to: {init_obs_path}")

def prepare_batch_recorders(camera_names, n_envs, instructions, video_test_dir: Path):
    """Allocate per-camera episode buffers and directories for video export."""
    batch_data = dict()
    video_dir_camera_dir = dict()
    for camera_name in camera_names:
        batch_data[camera_name] = create_batch_episode_data(num_envs=n_envs, instructions=instructions)
        video_dir_camera_dir[camera_name] = video_test_dir / f"{camera_name}"
        video_dir_camera_dir[camera_name].mkdir(parents=True, exist_ok=True)
    return batch_data, video_dir_camera_dir

def generate_videos_all_views(batch_data, video_dir_camera_dir, fps, n_envs):
    """Create per-camera episode videos from recorded batches."""
    custom_filename = f"{n_envs}envs"
    for camera_name in batch_data.keys():
        create_batch_videos(
            batch_data[camera_name],
            video_dir_camera_dir[camera_name],
            fps=fps,
            filter_success=False,
            custom_filename=custom_filename,
        )

def copy_hydra_config_to(destination_dir: Path):
    """Copy Hydra runtime configuration artifacts into a destination directory."""
    hydra_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    hydra_config_src = hydra_output_dir / ".hydra"
    if hydra_config_src.exists():
        hydra_dest = destination_dir / ".hydra"
        shutil.copytree(hydra_config_src, hydra_dest, dirs_exist_ok=True)
        print(f"✓ Copied Hydra config to {hydra_dest}")

def report_test_results(init_obs_dir, video_test_dir):
    print(f"\n{'='*60}")
    print(f"Test completed successfully!")
    print(f"  - Initial observation saved to: {init_obs_dir}")
    print(f"  - Video saved to: {video_test_dir}")
    print(f"  - Configs saved to both directories")
    print(f"{'='*60}")


def language_interface(n_envs:int=1)->list[str]:
    return [f"Test env {i}" for i in range(n_envs)]

def action_interface(obs:dict, n_envs:int, action_replan_horizon:int, single_action_dim:int, sim_device:torch.device)->torch.Tensor:
    return torch.randn(n_envs, action_replan_horizon, single_action_dim, device=sim_device)


@hydra.main(version_base=None, config_path="env_configs", config_name="stack_cubes")
def main(cfg: DictConfig):
    """Main function using Hydra configuration management."""
    
    # Print configuration
    print("=" * 60)
    print("Environment configuration loaded:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)
    
    cfg = flatten_config_if_nested(cfg)
    
    # Extract configuration values
    env_id = cfg.env_id
    env_type = cfg.env_type
    control_mode = cfg.control_mode
    episode_mode = cfg.episode_mode
    sim_config = OmegaConf.to_container(cfg.sim_config, resolve=True)
    sensor_config = OmegaConf.to_container(cfg.sensor_config, resolve=True)
    shader_pack = sensor_config.get("shader_pack", "default") if sensor_config else "default"
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
    action_replan_horizon = cfg.action_replan_horizon
    single_action_dim = cfg.single_action_dim
    obj_set = cfg.get("obj_set", None)
    use_render_camera = cfg.get("use_render_camera", False)  # Flag to use render() instead of obs from .step()
    print(f"\nCamera mode: {'render() camera' if use_render_camera else 'observation camera'}")

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
        shader_pack=shader_pack,  # NEW: Pass shader pack for camera configs
    )
    
    base_dir = script_dir.parent.parent / "log" / "quick_test_maniskill"
    timestamp, init_obs_dir, video_test_dir = prepare_output_dirs(base_dir, env_id)
    
    # Test env.reset() and save initial observations
    print(f"\nTest env.reset()")
    copy_hydra_config_to(init_obs_dir)
    obs, info = env.reset()
    reward = torch.zeros(n_envs, device=sim_device)
    print(f"obs={obs.keys()}, info={info.keys()}")
    obs_rgb_overlay = get_rgb_obs_with_info_overlay(
        env, obs, info, reward, env_type, sim_device, model_device, use_render_camera
    )
    for camera_name, img in obs_rgb_overlay.items():
        print(f"  Camera: {camera_name}, Image shape: {img.shape}")
    save_initial_observations_to_disk(obs_rgb_overlay, init_obs_dir)
    actions = action_interface(obs, n_envs, action_replan_horizon, single_action_dim, sim_device)
    
    # Run test episodes with video recording
    print(f"\nRunning {num_test_steps} test steps with video recording...")
    

    #########################################################
    # Create dummy instructions for each environment
    instructions = language_interface(n_envs)
    #########################################################

    # Create batch episode data recorder
    batch_data, video_dir_camera_dir = prepare_batch_recorders(
        obs_rgb_overlay.keys(), n_envs, instructions, video_test_dir
    )
    
    # Run episode loop
    for step in tqdm(range(num_test_steps)):
        #########################################################
        # Generate random actions for testing
        actions = action_interface(obs, n_envs, action_replan_horizon, single_action_dim, sim_device)
        
        #########################################################

        obs, reward, terminated, truncated, info = env.step(actions)
        
        # Record RGB observations with info overlay for each camera
        obs_rgb_overlay = get_rgb_obs_with_info_overlay(
            env, obs, info, reward, env_type, sim_device, model_device, use_render_camera
        )
        for camera_name in obs_rgb_overlay.keys():
            batch_data[camera_name].add_step_data_in_batch(
                obs_rgb_batch=obs_rgb_overlay[camera_name].permute(0, 2, 3, 1),   # [B, C, H, W] -> [B, H, W, C]
                rewards=reward,
                step_info=info,
                actions_batch=actions
            )
        
    
    print(f"\nCreating video...")
    copy_hydra_config_to(video_test_dir)
    generate_videos_all_views(batch_data, video_dir_camera_dir, frame_per_second, n_envs)
    report_test_results(init_obs_dir, video_test_dir)


if __name__ == "__main__":
    main()
