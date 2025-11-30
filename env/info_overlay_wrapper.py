"""
InfoOverlayWrapper - A gymnasium wrapper that automatically adds info overlays to rendered images.

This wrapper intercepts the render() call and automatically overlays info (success, rewards, etc.)
on the rendered images without requiring any changes to your rollout code.

Usage:
    from env.info_overlay_wrapper import InfoOverlayWrapper
    from functools import partial
    
    wrappers = [
        partial(PerStepRewardWrapper),
        partial(MultiActionWrapper),
        partial(InfoOverlayWrapper),  # Add this!
    ]
    
    env = setup_maniskill_env(..., wrappers=wrappers)
    
    # Now rendering automatically includes info overlays!
    # No code changes needed in your rollout
"""

import torch
import numpy as np
import gymnasium
from typing import Optional, Dict, Union
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, gym_utils
from mani_skill.utils.visualization.misc import put_info_on_image, tile_images


class InfoOverlayWrapper(gymnasium.Wrapper):
    """
    A wrapper that automatically adds info overlays to rendered images.
    
    This wrapper stores the current info and rewards from the last step/reset,
    and uses them to overlay information on rendered images automatically.
    
    Features:
    - Stores last info and rewards
    - Automatically overlays info on render()
    - No changes needed to rollout code
    - Works with both single env and vectorized envs
    """
    
    def __init__(self, env: BaseEnv, enable_overlay: bool = True):
        """
        Args:
            env: The ManiSkill environment to wrap
            enable_overlay: Whether to enable info overlay (default: True)
        """
        super().__init__(env)
        self.env: BaseEnv = env
        self.num_envs = env.unwrapped.num_envs
        self.enable_overlay = enable_overlay
        
        # Store current info and rewards for rendering
        self._current_info = None
        self._current_reward = None
        
        self.debug_mode = False
    
    def reset(self, seed=None, options: dict = {}) -> tuple[dict, dict]:
        """Reset environment and store initial info."""
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Store info for rendering
        self._current_info = info
        self._current_reward = None  # No reward at reset
        
        return obs, info
    
    def step(self, action):
        """Step environment and store info and rewards."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Store info and rewards for rendering
        self._current_info = info
        self._current_reward = reward
        
        if self.debug_mode:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"DEBUG::{self.__class__.__name__}: Stored info and rewards for rendering")
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """
        Override render to add info overlays.
        
        This methorendere base env's render() to get images
        2. Extracts scalar info from stored info dict
        3. Overlays the info on each image
        4. Returns the overlayed images
        """
        # Get base rendered images
        img = self.env.render()
        
        # If overlay is disabled or no info available, return original
        if not self.enable_overlay or self._current_info is None:
            return img
        
        # Convert to numpy
        img = common.to_numpy(img)
        if len(img.shape) == 3:
            img = img[None]  # Add batch dimension if single image
        
        # Extract scalar info
        scalar_info = gym_utils.extract_scalars_from_info(
            common.to_numpy(self._current_info), 
            batch_size=self.num_envs
        )
        
        # Add rewards to scalar info if available
        if self._current_reward is not None:
            scalar_info["reward"] = common.to_numpy(self._current_reward)
            if np.size(scalar_info["reward"]) > 1:
                scalar_info["reward"] = [
                    float(rew) for rew in scalar_info["reward"]
                ]
            else:
                scalar_info["reward"] = float(scalar_info["reward"])
        
        # Overlay info on each image
        overlayed_imgs = []
        for i in range(len(img)):
            info_item = {
                k: v if np.size(v) == 1 else v[i] 
                for k, v in scalar_info.items()
            }
            overlayed_img = put_info_on_image(img[i], info_item)
            overlayed_imgs.append(overlayed_img)
        
        # Stack back to batch
        result = np.stack(overlayed_imgs, axis=0)
        
        # Remove batch dimension if originally single image
        if result.shape[0] == 1:
            result = result[0]
        
        return result
    
    def close(self):
        """Close the environment."""
        return self.env.close()


class InfoOverlayWrapperWithCapture(InfoOverlayWrapper):
    """
    Extended version that also provides methods to capture overlayed images from observations.
    
    This is useful when you want to get overlayed images from observations
    (e.g., from sensor_data) rather than from render().
    """
    
    def __init__(self, env: BaseEnv, enable_overlay: bool = True):
        super().__init__(env, enable_overlay)
    
    def get_overlayed_images_from_obs(
        self, 
        obs: Dict,
        camera_name: str = "3rd_view_camera",
        tile: bool = False
    ) -> np.ndarray:
        """
        Extract images from observation and add info overlays.
        
        Args:
            obs: Observation dictionary containing sensor_data
            camera_name: Name of the camera to extract images from
            tile: Whether to tile images into a grid (for multi-env)
            
        Returns:
            Overlayed images as numpy array
                - If tile=False: (B, H, W, C)
                - If tile=True: (H_tiled, W_tiled, C)
        """
        if not self.enable_overlay or self._current_info is None:
            # Just extract and return images without overlay
            imgs = obs["sensor_data"][camera_name]["rgb"]
            imgs = common.to_numpy(imgs)
            if tile and len(imgs) > 1:
                return tile_images(imgs, nrows=int(np.sqrt(self.num_envs)))
            return imgs
        
        # Extract RGB images from observation
        imgs = obs["sensor_data"][camera_name]["rgb"]  # (B, H, W, C)
        imgs = common.to_numpy(imgs)
        
        # Extract scalar info
        scalar_info = gym_utils.extract_scalars_from_info(
            common.to_numpy(self._current_info), 
            batch_size=self.num_envs
        )
        
        # Add rewards to scalar info if available
        if self._current_reward is not None:
            scalar_info["reward"] = common.to_numpy(self._current_reward)
            if np.size(scalar_info["reward"]) > 1:
                scalar_info["reward"] = [
                    float(rew) for rew in scalar_info["reward"]
                ]
            else:
                scalar_info["reward"] = float(scalar_info["reward"])
        
        # Overlay info on each image
        overlayed_imgs = []
        for i in range(len(imgs)):
            info_item = {
                k: v if np.size(v) == 1 else v[i] 
                for k, v in scalar_info.items()
            }
            overlayed_img = put_info_on_image(imgs[i], info_item)
            overlayed_imgs.append(overlayed_img)
        
        overlayed_imgs = np.stack(overlayed_imgs, axis=0)
        
        # Tile if requested
        if tile and len(overlayed_imgs) > 1:
            return tile_images(overlayed_imgs, nrows=int(np.sqrt(self.num_envs)))
        
        return overlayed_imgs


# For backwards compatibility
InfoRenderWrapper = InfoOverlayWrapper


if __name__ == "__main__":
    print("InfoOverlayWrapper - Gymnasium wrapper for automatic info overlays")
    print("=" * 70)
    print("""
Usage Example:

from functools import partial
from env.info_overlay_wrapper import InfoOverlayWrapper

# Add to your wrappers list
wrappers = [
    partial(PerStepRewardWrapper),
    partial(MultiActionWrapper),
    partial(InfoOverlayWrapper),  # <-- Add this!
]

# Setup environment
env = setup_maniskill_env(
    env_id, n_envs, n_steps_episode, 
    sim_backend, sim_device, sim_device_id,
    sim_config, sensor_config, obs_mode, control_mode,
    episode_mode, obj_set,
    wrappers=wrappers  # <-- Pass wrappers
)

# Now your rollout code stays the same!
obs, info = env.reset()
for step in range(num_steps):
    actions = policy(obs)
    obs, reward, terminated, truncated, info = env.step(actions)
    
    # Render automatically includes info overlays!
    img = env.render()  # <-- Info overlayed automatically!
    
# No code changes needed in rollout - wrapper handles everything!
    """)
