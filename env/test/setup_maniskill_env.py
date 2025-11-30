
import torch 
from ManiSkill.mani_skill.envs.sapien_env import BaseEnv
from omegaconf import DictConfig, OmegaConf
import gymnasium as gym
from gymnasium.core import Env
from ManiSkill.mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
import logging
logger=logging.getLogger(__name__)

def setup_maniskill_env(env_id, 
                        n_envs, 
                        max_episode_len,
                        sim_backend, 
                        sim_device, 
                        sim_device_id,
                        sim_config,
                        sensor_config,
                        obs_mode,
                        render_mode,
                        control_mode,
                        episode_mode='eval',
                        obj_set=None,
                        wrappers=None, # list of wrappers to wrap the environment. 
                        )->ManiSkillVectorEnv:
    """
    Args:
        env_id: str, the id of the environment to setup.
        num_envs: int, the number of environments to setup.
        max_episode_len: int, the maximum number of steps per episode.
        sim_backend: str, the backend to use for simulation.
        sim_device: str, the device to use for simulation.
        sim_device_id: int, the id of the device to use for simulation.
        sim_config: dict, the configuration for simulation.
        sensor_config: dict, the configuration for sensors.
        obs_mode: str, the mode to use for observations.
        control_mode: str, the mode to use for control.
        episode_mode: str, the mode to use for episodes.
        obj_set: str, the set of objects to use for the environment.
        wrappers: list, the wrappers to use for the environment.
    Returns:
        venv: ManiSkillVectorEnv, the vectorized environment.
    
    if mode=='eval':
            Parallel evaluation, run only one episode per environment, no reset within. 
    elif mode=='train':
            Parallel RL training, run multiple episodes per environment, each environment automatically resets and does not interfere with others. 
    """
    # Setup ManiSkill3 environment
    logger.info(f"Setting up ManiSkill3 environment: {env_id}")
    
    # Convert OmegaConf objects to regular Python dictionaries to avoid compatibility issues
    for tmp_config in [sim_config, sensor_config]:
        if isinstance(tmp_config, DictConfig):
            tmp_config = OmegaConf.to_container(tmp_config, resolve=True)


    # Determine autoreset. 
    if episode_mode=='eval':
        # Parallel evaluation, run only one episode per environment, no reset within. 
        auto_reset=False
        ignore_terminations=True
        reconfiguration_freq=1
    elif episode_mode=='train':
        # Parallel RL training, run multiple episodes per environment, each environment automatically resets and does not interfere with others. 
        auto_reset=True
        ignore_terminations=False
        reconfiguration_freq=0
    else:
        raise NotImplementedError(f"Episode mode {episode_mode} not implemened, we only support 'eval' and 'train'.")
    try:
        if sim_backend == "gpu":
            logger.info(f"Setting up simulation on CUDA device: {sim_device}")
            logger.info(f"About to call torch.cuda.set_device({sim_device_id}). Current CUDA device before set_device: {torch.cuda.current_device()}")
            torch.cuda.set_device(sim_device_id)            
            logger.info(f"Current CUDA device after set_device: {torch.cuda.current_device()}")
            logger.info(f"Available devices: {torch.cuda.device_count()}")
            
            if obj_set is not None:
                env: BaseEnv = gym.make( # type: ignore
                id=env_id,
                num_envs=n_envs,
                max_episode_steps=max_episode_len,
                obs_mode=obs_mode,
                render_mode=render_mode,
                control_mode=control_mode,
                sim_backend=sim_backend,
                sim_config=sim_config,
                sensor_configs=sensor_config,
                viewer_camera_configs=dict(shader_pack=sensor_config.get("shader_pack", "default")),
                reconfiguration_freq=reconfiguration_freq,
                obj_set=obj_set
            )# type: ignore
            else:
                env: BaseEnv = gym.make( # type: ignore
                    id=env_id,
                    num_envs=n_envs,
                    max_episode_steps=max_episode_len,
                    obs_mode=obs_mode,
                    render_mode=render_mode,
                    control_mode=control_mode,
                    sim_backend=sim_backend,
                    sim_config=sim_config,
                    sensor_configs=sensor_config,
                    viewer_camera_configs=dict(shader_pack=sensor_config.get("shader_pack", "default")),
                    reconfiguration_freq=reconfiguration_freq,
                )# type: ignore
            
            # Wrap the environment with the wrappers, if provided. 
            if wrappers:
                logger.info(f"Wrapping up BaseEnv with custom wrappers:")
                for wrapper in wrappers:
                    logger.info(f"\tWrapping up BaseEnv with custom wrapper {wrapper.func.__name__}")
                    env = wrapper(env) 
            logger.info(f"Wrapping environment with ManiSkillVectorEnv with auto_reset={auto_reset} and ignore_terminations={ignore_terminations}.")
            venv: ManiSkillVectorEnv=ManiSkillVectorEnv(env,
                                                        auto_reset= auto_reset,
                                                        ignore_terminations= ignore_terminations)
            logger.info(f"Double checking: env.base_env.reconfiguration_freq={venv.base_env.reconfiguration_freq}, env.auto_reset={venv.auto_reset} env.ignore_terminations={venv.ignore_terminations}")
        else:
            raise NotImplementedError
    except Exception as e:
        logger.info(f"Error setting up ManiSkill environment: {e}")
        raise
    logger.info(f"Successfully setup ManiSkill3 environment {env} on sim_device={sim_device}")
    
    env_unwrapped: Env=venv.unwrapped
    logger.info(f"Sim environment details:")
    env_unwrapped.print_sim_details()
    return venv # type: ignore