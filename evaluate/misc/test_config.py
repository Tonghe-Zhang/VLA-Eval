#!/usr/bin/env python3
"""
Test script to verify Hydra configuration loading for pi0 ManiSkill3 evaluation.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

@hydra.main(version_base=None, config_path="config", config_name="default")
def test_config(cfg: DictConfig):
    """Test the configuration loading"""
    
    print("Testing Hydra Configuration Loading")
    print("=" * 50)
    
    # Print the configuration
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Test specific configuration sections
    print("\nEnvironment Configuration:")
    print(f"  ID: {cfg.env.id}")
    print(f"  Num envs: {cfg.env.num_envs}")
    print(f"  Max episode len: {cfg.env.max_episode_len}")
    
    print("\nModel Configuration:")
    print(f"  Path: {cfg.model.path}")
    print(f"  Device: {cfg.model.device}")
    
    print("\nEvaluation Configuration:")
    print(f"  Num episodes: {cfg.eval.num_episodes}")
    print(f"  Seed: {cfg.eval.seed}")
    
    print("\nOutput Configuration:")
    print(f"  Directory: {cfg.output.dir}")
    print(f"  Save videos: {cfg.output.save_videos}")
    print(f"  Save data: {cfg.output.save_data}")
    
    print("\nConfiguration test completed successfully!")

if __name__ == "__main__":
    test_config() 