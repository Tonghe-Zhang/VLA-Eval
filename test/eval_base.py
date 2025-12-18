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
import sys
from functools import partial
import json
import numpy as np

# Set up paths
script_dir = Path(__file__).parent  # test/
quick_test_dir = script_dir.parent  # quick_test_maniskill/
manip_dir = quick_test_dir.parent  # Manip/
project_root = manip_dir.parent  # Project/
maniskill_root = manip_dir / "ManiSkill"

# Add paths to sys.path
sys.path.insert(0, str(maniskill_root))
sys.path.insert(0, str(quick_test_dir))

# Add openpi path if it exists
openpi_src = project_root / "openpi-main" / "src"
if openpi_src.exists():
    sys.path.insert(0, str(openpi_src))

# Import custom environment definitions
try:
    import SimplerEnvPlus.tasks.put_carrot_on_plate
    import SimplerEnvPlus.tasks.put_on_in_scene_multi
    import SimplerEnvPlus.tasks.variants
    print("Successfully registered custom SimplerEnvPlus environments")
except ImportError as e:
    print(f"Warning: Could not import SimplerEnvPlus tasks: {e}")

# Local imports
from ManiSkill.mani_skill.utils import common
# For FK solver
from ManiSkill.mani_skill.utils.structs.articulation import Articulation
from ManiSkill.mani_skill.agents.controllers.utils.kinematics import Kinematics
from ManiSkill.mani_skill.agents.utils import get_active_joint_indices
# Env and its wrappers
from quick_test_maniskill.test.setup_maniskill_env import setup_maniskill_env
from ManiSkill.mani_skill.envs.sapien_env import BaseEnv  
from ManiSkill.mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from quick_test_maniskill.env.multi_action_wrapper import MultiActionWrapper
from quick_test_maniskill.env.per_step_reward_wrapper import PerStepRewardWrapper
from quick_test_maniskill.env.fetch_rgb_from_obs import fetch_rgb_from_obs_allenvs, overlay_info_on_rgb_image
from quick_test_maniskill.evaluate.eval_helpers_versatile import (
    create_batch_episode_data, 
    create_batch_videos,
    tile_images
)

from quick_test_maniskill.env.forward_kinematics import forward_kinematics
from SimplerEnvPlus.data_generation.maniskill_custom_package.controller_utils.kinematics import Kinematics






class BaseEnvTester:
    """
    Base class for testing ManiSkill environments.
    Handles environment setup, loop, data recording, and visualization.
    """
    def __init__(self, cfg: DictConfig):
        self.cfg = self.flatten_config_if_nested(cfg)
        self.setup_config()
        self.setup_directories()
        self.setup_env()
        
    def flatten_config_if_nested(self, cfg: DictConfig) -> DictConfig:
        """Flatten Hydra configs that are wrapped inside group names."""
        # Recursively flatten until we reach a config with env_id or env_type at the top level
        while len(cfg) == 1 and not any(k in cfg for k in ['env_id', 'env_type']):
            group_name = list(cfg.keys())[0]
            print(f"Flattening nested config group: {group_name}")
            cfg = cfg[group_name]
        return cfg

    def setup_config(self):
        """Extract configuration values."""
        print(f"self.cfg={self.cfg}")
        self.env_id = self.cfg.env_id
        self.env_type = self.cfg.env_type
        self.control_mode = self.cfg.control_mode
        self.episode_mode = self.cfg.episode_mode
        self.sim_config = OmegaConf.to_container(self.cfg.sim_config, resolve=True)
        self.sensor_config = OmegaConf.to_container(self.cfg.sensor_config, resolve=True)
        self.shader_pack = self.sensor_config.get("shader_pack", "default") if self.sensor_config else "default"
        self.obs_mode = self.cfg.obs_mode
        self.n_envs = self.cfg.n_envs
        self.proprioception_type = self.cfg.proprioception_type # ee_pose, qpos
        self.n_steps_episode = self.cfg.n_steps_episode
        self.num_test_steps = self.cfg.num_test_steps
        self.frame_per_second = self.cfg.frame_per_second
        self.sim_backend = self.cfg.sim_backend
        self.sim_device = self.cfg.sim_device
        self.sim_device_id = self.cfg.sim_device_id
        self.model_device = self.cfg.model_device
        self.render_mode = self.cfg.render_mode
        self.action_replan_horizon = self.cfg.action_replan_horizon
        self.single_action_dim = self.cfg.single_action_dim
        self.obj_set = self.cfg.get("obj_set", None)
        self.use_render_camera = self.cfg.get("use_render_camera", False)
        self.save_detailed_logs = self.cfg.get("save_detailed_logs", False)  # Default: don't save verbose step_infos
        self.show_success_overlay = self.cfg.get("show_success_overlay", True)  # Default: show green overlay for successful envs
        
        print(f"\nCamera mode: {'render() camera' if self.use_render_camera else 'observation camera'}")
        print(f"Detailed logging (step_infos): {'enabled' if self.save_detailed_logs else 'disabled (saves space)'}")
        print(f"Success overlay: {'enabled (green mask on successful envs)' if self.show_success_overlay else 'disabled'}")

    def setup_directories(self):
        """Prepare output directories."""
        base_dir = script_dir.parent.parent / "log" / "quick_test_maniskill"
        timestamp = datetime.now().strftime("%Y-%d-%m--%H-%M-%S")
        self.init_obs_dir = base_dir / "figs" / "init-obs" / f"{self.env_id}" / f"{timestamp}"
        self.video_test_dir = base_dir / "video_test" / f"{self.env_id}" / f"{timestamp}"
        
        self.init_obs_dir.mkdir(parents=True, exist_ok=True)
        self.video_test_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Output directories created:")
        print(f"  Init obs dir: {self.init_obs_dir}")
        print(f"  Video dir: {self.video_test_dir}")
        
        self.copy_hydra_config_to(self.init_obs_dir)
        self.copy_hydra_config_to(self.video_test_dir)

    def copy_hydra_config_to(self, destination_dir: Path):
        """Copy Hydra runtime configuration artifacts."""
        try:
            hydra_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
            hydra_config_src = hydra_output_dir / ".hydra"
            if hydra_config_src.exists():
                hydra_dest = destination_dir / ".hydra"
                shutil.copytree(hydra_config_src, hydra_dest, dirs_exist_ok=True)
                print(f"✓ Copied Hydra config to {hydra_dest}")
        except Exception as e:
            print(f"Could not copy hydra config (might not be running via hydra): {e}")

    def setup_env(self):
        """Initialize the environment."""
        wrappers = [
            partial(PerStepRewardWrapper),
            partial(MultiActionWrapper),
        ]
        
        self.env: ManiSkillVectorEnv = setup_maniskill_env(
            env_id=self.env_id, 
            n_envs=self.n_envs, 
            max_episode_len=self.n_steps_episode, 
            sim_backend=self.sim_backend, 
            sim_device=self.sim_device, 
            sim_device_id=self.sim_device_id, 
            sim_config=self.sim_config, 
            sensor_config=self.sensor_config, 
            obs_mode=self.obs_mode, 
            render_mode=self.render_mode,
            control_mode=self.control_mode, 
            episode_mode=self.episode_mode, 
            obj_set=self.obj_set,
            wrappers=wrappers,
            shader_pack=self.shader_pack,
        )

    def get_rgb_obs_with_info_overlay(self, obs, info, reward, language_instruction=None):
        """Get RGB observations with info overlay."""
        if self.use_render_camera: # obtained from env.render()
            rendered_img = self.env.render()
            rendered_img = common.to_numpy(rendered_img)
            if len(rendered_img.shape) == 3:
                rendered_img = rendered_img[None]
            
            rendered_img_torch = torch.from_numpy(rendered_img).to(self.sim_device)
            rendered_img_torch = rendered_img_torch.permute(0, 3, 1, 2)
            
            obs_rgb_overlay = {"render_camera": rendered_img_torch}
            obs_rgb_overlay["render_camera"] = overlay_info_on_rgb_image(
                obs_rgb_overlay["render_camera"], info, reward, language_instruction
            )
        else:
            # Use observation camera obtained from env.step()
            obs_rgb_overlay = fetch_rgb_from_obs_allenvs(
                self.env_type, obs, self.sim_device, self.model_device, info=info, reward=reward, language_instruction=language_instruction, normalize=False
            )
        return obs_rgb_overlay

    def save_initial_observations(self, obs_rgb_overlay):
        """Save initial observations to disk."""
        for camera_name, img in obs_rgb_overlay.items():
            img_tiled = img.permute(0, 2, 3, 1).cpu().numpy()   
            tiled_init_obs = tile_images(img_tiled)   # [B, H, W, C] 
            init_obs_path = self.init_obs_dir / f"{camera_name}.png"
            Image.fromarray(tiled_init_obs).save(init_obs_path)
            print(f"Saved initial observation image to: {init_obs_path}")

    def _make_json_serializable(self, obj):
        """Recursively convert objects to JSON-serializable types."""
        import torch
        import numpy as np
        
        if isinstance(obj, (torch.Tensor, np.ndarray)):
            # Convert tensors/arrays to lists
            if hasattr(obj, 'cpu'):
                obj = obj.cpu()
            if hasattr(obj, 'numpy'):
                obj = obj.numpy()
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            return obj
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            # Try to convert to string as fallback
            return str(obj)
    
    def report_results(self, batch_data_all_cameras):
        """Report results including success statistics and save logs."""
        # Calculate success statistics across all environments
        success_once_list = []
        success_at_end_list = []
        
        for camera_name, batch_data in batch_data_all_cameras.items():
            # Check for different success metrics
            if hasattr(batch_data, 'success_once'):
                success_once_list = batch_data.success_once
            if hasattr(batch_data, 'success_at_end'):
                success_at_end_list = batch_data.success_at_end
            elif hasattr(batch_data, 'success'):
                # Fallback to generic 'success' if specific ones not available
                success_at_end_list = batch_data.success
            else:
                raise ValueError(f"No success metrics found in batch data for camera: {camera_name}")
            
            if success_once_list or success_at_end_list:
                break
        
        # Print success statistics if available
        has_success_metrics = False
        if success_once_list or success_at_end_list:
            print(f"\n{'='*60}")
            print(f"SUCCESS STATISTICS:")
            
            if success_once_list:
                success_once_rate = np.mean(success_once_list)
                num_success_once = int(np.sum(success_once_list))
                print(f"  - Success Once: {success_once_rate:.2%} ({num_success_once}/{len(success_once_list)} envs)")
                has_success_metrics = True
            
            if success_at_end_list:
                success_at_end_rate = np.mean(success_at_end_list)
                num_success_at_end = int(np.sum(success_at_end_list))
                print(f"  - Success At End: {success_at_end_rate:.2%} ({num_success_at_end}/{len(success_at_end_list)} envs)")
                has_success_metrics = True
            
            print(f"{'='*60}")
        
        # Save evaluation logs (excluding video images) to JSON
        log_data = {}
        for camera_name, batch_data in batch_data_all_cameras.items():
            # Always save: instructions, success, rewards, actions, proprioception
            camera_log = {
                "num_envs": batch_data.num_envs,
                "instructions": batch_data.instructions if hasattr(batch_data, 'instructions') else [],
                "rewards": [[float(r) for r in env_rewards] for env_rewards in batch_data.rewards],
                "actions": [[a.tolist() if hasattr(a, 'tolist') else a for a in env_actions] for env_actions in batch_data.actions],
                "proprioception": [[p.tolist() if hasattr(p, 'tolist') else p for p in env_prop] for env_prop in batch_data.proprioception],
            }
            
            # Add success metrics if available
            if hasattr(batch_data, 'success_once'):
                camera_log["success_once"] = [bool(s) for s in batch_data.success_once]
            if hasattr(batch_data, 'success_at_end'):
                camera_log["success_at_end"] = [bool(s) for s in batch_data.success_at_end]
            elif hasattr(batch_data, 'success'):
                camera_log["success"] = [bool(s) for s in batch_data.success]
            
            # Only save verbose step_infos if enabled (this is what makes files HUGE!)
            if self.save_detailed_logs:
                camera_log["step_infos"] = self._make_json_serializable(batch_data.infos) if hasattr(batch_data, 'infos') else []
            
            log_data[camera_name] = camera_log
        
        # Add summary statistics
        if has_success_metrics:
            summary = {}
            if success_once_list:
                summary["success_once_rate"] = float(np.mean(success_once_list))
                summary["num_success_once"] = int(np.sum(success_once_list))
            if success_at_end_list:
                summary["success_at_end_rate"] = float(np.mean(success_at_end_list))
                summary["num_success_at_end"] = int(np.sum(success_at_end_list))
            summary["num_total"] = len(success_once_list) if success_once_list else len(success_at_end_list)
            log_data["summary"] = summary
        
        # Save to JSON file in video directory
        log_path = self.video_test_dir / "eval_logs.json"
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Test completed successfully!")
        print(f"  - Initial observation saved to: {self.init_obs_dir}")
        print(f"  - Video saved to: {self.video_test_dir}")
        print(f"  - Evaluation logs saved to: {log_path}")
        print(f"{'='*60}")
    
    # --- Methods to be overridden by subclasses ---

    def get_language_instruction(self, n_envs: int) -> list[str]:
        """Get language instructions for the batch."""
        return [f"Test env {i}" for i in range(n_envs)]

    def get_proprioception(self, env, prop_type: str = "qpos") -> torch.Tensor:
        """
        Get proprioception from the environment.
        Args:
            env: ManiSkill environment
            prop_type: Type of proprioception to get. "qpos", "qvel", "state", "ee_pose_UMI", "ee_pose". 
            * When we use "qpos", we get the joint positions of all active joints. [B, 8] for a 6 dof arm with 2 grippers. That quantity contains the states for each gripper. 
            * When we use "ee_pose", by default (FK) we only get the [position, quaternion]=7 dimension pose of the end-effector in the robot base frame, which excludes gripper states. 
              We also consider the error in that measurement so we either use FK or SLAM, and don't query the absolte state obtained by the simulator. 
            * Notice that while action are usually 7 dim for a single 2-gripper arm, [position, euler_angle, gripper_state], we use quaternion in ee_pose state. You need a conversion to make them euler angles. 
        Returns:
            proprioception: Proprioception tensor [B, dim]
        """

        if hasattr(env.unwrapped, 'agent') and hasattr(env.unwrapped.agent, 'robot'):
            env_unwrapped: BaseEnv = env.unwrapped
            agent = env_unwrapped.agent
            robot: Articulation = env_unwrapped.agent.robot
            if prop_type == "qpos":  # joint positions, dim=joint_dof, which is 6 + 2 = 8 for the arm joints + 2 for the gripper joints as of WidowX250S. 
                proprioception = robot.get_qpos()
            elif prop_type == "qvel":  # joint velocities, dim=joint_dof
                proprioception = robot.get_qvel()
            elif prop_type == "state":  # [pose.p, pose.q, vel, ang_vel, qpos, qvel], dim=7+3+3+2*joint_dof
                proprioception = robot.get_state()
            elif prop_type == "ee_pose":  # [pose.p, pose.q], dim=7 x num_arms. Standard FK solution, position + quaternion of end-effector. 
                qpos = robot.get_qpos()   # all active joint positoins. [B, 8] for a 6 dof arm with 2 grippers. 
                arm_joint_indices = get_active_joint_indices(robot, agent.arm_joint_names)
                qpos_arm = qpos[:, arm_joint_indices]
                kinematics = Kinematics(
                    urdf_path=agent.urdf_path,
                    end_link_name=agent.ee_link_name, #"ee_gripper_link",
                    articulation=robot,
                    active_joint_indices=arm_joint_indices
                )
                # Only pass the qpos for the arm joints (6 DOF) instead of full robot qpos (8 DOF includes 2 dims for the 2 grippers, gripper_joint_names = ["left_finger", "right_finger"])
                ee_pose = forward_kinematics(env_unwrapped, kinematics, qpos_arm, world_frame=False)
                proprioception = torch.hstack([ee_pose.p, ee_pose.q])
            elif prop_type == "ee_pose_UMI":  # [pose.p, pose.q], dim=7 x num_arms. This one requires visual SLAM. 
                raise NotImplementedError("ee_pose_UMI is not implemented")  
            else:
                raise ValueError(f"Invalid proprioception type: {prop_type}")
            return proprioception.to(self.model_device) # [B, dim]
        else: 
            raise ValueError("env.unwrapped.agent.robot not found")

    
    def get_action(self, obs, proprioception, language_instruction) -> torch.Tensor:
        """Get action for the current observation."""
        return torch.randn(self.n_envs, self.action_replan_horizon, self.single_action_dim, device=self.sim_device)

    def check_success(self, info, env_idx: int) -> bool:
        """Check if environment at env_idx is successful."""
        # Default success check logic (can be expanded)
        if 'success' in info:
             val = info['success']
             if hasattr(val, '__getitem__') and len(val) > env_idx:
                 return val[env_idx]
        return False

    # ----------------------------------------------

    def run(self):
        """Main execution loop."""
        print(f"\nTest env.reset()")
        obs, info = self.env.reset()
        reward = torch.zeros(self.n_envs, device=self.sim_device)
        language_instruction = self.get_language_instruction(self.n_envs)
        
        # Initial observation
        obs_rgb_overlay = self.get_rgb_obs_with_info_overlay(obs, info, reward, language_instruction)
        self.save_initial_observations(obs_rgb_overlay)
        print(f"\nRunning {self.num_test_steps} test steps with video recording...")
        
        
        
        # Prepare recorders
        batch_data = dict()
        video_dir_camera_dir = dict()
        for camera_name in obs_rgb_overlay.keys():
            batch_data[camera_name] = create_batch_episode_data(num_envs=self.n_envs)
            video_dir_camera_dir[camera_name] = self.video_test_dir / f"{camera_name}"
            video_dir_camera_dir[camera_name].mkdir(parents=True, exist_ok=True)

        # Step loop
        for step in tqdm(range(self.num_test_steps)):
            proprioception = self.get_proprioception(self.env, prop_type=self.proprioception_type)
            
            actions = self.get_action(obs, proprioception, language_instruction)
            
            obs, reward, terminated, truncated, info = self.env.step(actions)
            
            obs_rgb_overlay = self.get_rgb_obs_with_info_overlay(obs, info, reward, language_instruction)

            for camera_name in obs_rgb_overlay.keys():
                batch_data[camera_name].add_step_data_in_batch(
                    obs_rgb_batch=obs_rgb_overlay[camera_name].permute(0, 2, 3, 1),
                    rewards=reward,
                    step_info=info,
                    actions_batch=actions,
                    proprioception=proprioception,
                    instruction=language_instruction
                )
        
        # Finalize success status for all cameras
        for camera_name in batch_data.keys():
            batch_data[camera_name].finalize_success_status()

        # Generate videos
        print(f"\nCreating video...")
        custom_filename = f"{self.n_envs}envs"
        for camera_name in batch_data.keys():
            create_batch_videos(
                batch_data[camera_name],
                video_dir_camera_dir[camera_name],
                fps=self.frame_per_second,
                filter_success=self.show_success_overlay,
                custom_filename=custom_filename,
            )
            
        self.report_results(batch_data)

@hydra.main(version_base=None, config_path="env_cfg", config_name="stack_cubes")
def main(cfg: DictConfig):
    print("Using BaseEnvTester (Random Actions)...")
    tester = BaseEnvTester(cfg)      
    tester.run()

if __name__ == "__main__":
    main()

