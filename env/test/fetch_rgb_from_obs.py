
import torch
from typing import Dict

from ManiSkill.mani_skill.utils import common, gym_utils
from ManiSkill.mani_skill.utils.visualization.misc import (
    put_info_on_image,
)
import numpy as np


def overlay_info_on_rgb_image(rgb_image: torch.Tensor, info: dict, reward: torch.Tensor) -> torch.Tensor:
    """
    Add info overlays to RGB images.
    
    Args:
        rgb_image: Torch tensor of shape [N, C, H, W] in uint8
        info: Info dict from environment
        reward: Reward tensor of shape [N]
    
    Returns:
        Overlayed images as torch tensor [N, C, H, W]
    """
    # Store original device and format
    device = rgb_image.device
    is_torch = isinstance(rgb_image, torch.Tensor)
    
    # Convert to numpy and to [N, H, W, C] format for put_info_on_image
    if is_torch:
        rgb_image_np = rgb_image.permute(0, 2, 3, 1).cpu().numpy()  # [N,C,H,W] -> [N,H,W,C]
    else:
        rgb_image_np = rgb_image
    
    # Extract scalar info
    scalar_info = gym_utils.extract_scalars_from_info(
        common.to_numpy(info), batch_size=rgb_image_np.shape[0]
    )
    
    if reward is not None:
        scalar_info["reward"] = common.to_numpy(reward)
        if np.size(scalar_info["reward"]) > 1:
            scalar_info["reward"] = [
                float(rew) for rew in scalar_info["reward"]
            ]
        else:
            scalar_info["reward"] = float(scalar_info["reward"])
    
    # Overlay info on each image (now in [H,W,C] format)
    overlayed_imgs = []
    for i in range(len(rgb_image_np)):
        info_item = {
            k: v if np.size(v) == 1 else v[i] for k, v in scalar_info.items()
        }
        overlayed_img = put_info_on_image(rgb_image_np[i], info_item)  # Input/output: [H,W,C]
        overlayed_imgs.append(overlayed_img)
    
    overlayed_imgs = np.stack(overlayed_imgs, axis=0)  # [N,H,W,C]
    
    # Convert back to torch and [N,C,H,W] format
    if is_torch:
        overlayed_imgs = torch.from_numpy(overlayed_imgs).permute(0, 3, 1, 2).to(device)  # [N,H,W,C] -> [N,C,H,W]
    
    return overlayed_imgs


def fetch_rgb_from_obs_allenvs(env_type: str, obs: dict, sim_device: str, model_device: str, info=None, reward=None, normalize: bool = False) -> Dict[str, torch.Tensor]:
    """
    This function extracts the rgb component from the full raw observation from ManiSkill3
    and converts the shape and device as needed.
    
    Args:
        env_type: Type of environment configuration
        obs: Raw observation dictionary from ManiSkill environment
        sim_device: Device where simulation is running (e.g., "cuda", "cpu")
        model_device: Device where model will run (e.g., "cuda", "cpu")
        normalize: If True, convert uint8 [0, 255] to float [0, 1] for model input.
                   If False (default), keep as uint8 for visualization/saving.
    
    Returns:
        img_dict: Dict[str, torch.Tensor]
            Dictionary mapping camera names to RGB images of shape [B, C, H, W]
            Images are uint8 [0, 255] by default, or float [0, 1] if normalize=True
    
    Supported env_types:
        - "simplerenv": Uses 3rd_view_camera
        - "pick-and-place-randomize": Uses 3rd_view_camera
        - "tabletop-gripper": Uses base_camera (for all Panda manipulation tasks)
    """
    
    if env_type in ["simplerenv", "pick-and-place-randomize"]:
        # SimplerEnv and randomized pick-and-place use 3rd person view camera
        rgb_image = obs["sensor_data"]["3rd_view_camera"]["rgb"].permute(0, 3, 1, 2)    # [N,H,W,C] -> [N,C,H,W]
        img_dict = {
            "3rd_view_camera": rgb_image,
        }
    elif env_type in ["tabletop-gripper"]:
        # Tabletop tasks (PickCube, StackCube, PushCube, etc.) use base_camera
        rgb_image = obs["sensor_data"]["base_camera"]["rgb"].permute(0, 3, 1, 2)        # [N,H,W,C] -> [N,C,H,W]
        img_dict = {
            "base_camera": rgb_image,
        }
    else:
        raise NotImplementedError(
            f"env_type '{env_type}' is not supported. "
            f"Supported types: 'simplerenv', 'pick-and-place-randomize', 'tabletop-gripper'"
        )
    
    # Process images: optionally normalize and transfer to model device
    processed_dict = {}
    for camera_name, rgb_image in img_dict.items():
        # Overlay info on each image if provided (before normalization)
        if info is not None:
            rgb_image = overlay_info_on_rgb_image(rgb_image, info, reward)
        
        # For model training: Convert to float and normalize to [0, 1] if requested (for model input)
        if normalize and rgb_image.dtype == torch.uint8:
            rgb_image = rgb_image.float() / 255.0
        
        # Transfer image to model device if needed
        if sim_device != model_device:
            rgb_image = rgb_image.to(model_device)
        
        processed_dict[camera_name] = rgb_image
    
    return processed_dict
    

