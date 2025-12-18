#!/bin/bash
# Clean script to evaluate OpenPI models on ManiSkill environments
# Uses Hydra for unified configuration management
#
# Usage: bash eval_openpi.sh <env_config> <model_config> [hydra_overrides...]
#
# Arguments:
#   composed_config - Combined config from configs/ (e.g., stackcube_pi0, carrot_pi05)
#   hydra_overrides - (Optional) Any Hydra config overrides (e.g., n_envs=8 model_cfg.pytorch_weight_path=/path)
#
# ============================================================================
# USAGE EXAMPLES
# ============================================================================
#
# Example 1: Basic usage
#   bash eval_openpi.sh stackcube_pi0
#
# Example 2: With checkpoint
#   bash eval_openpi.sh carrot_pi05 \
#     model_cfg.pytorch_weight_path=/home/tonghe/Project/PretrainedModels/pi05_base
#
# Example 3: Override env and model params
#   bash eval_openpi.sh stackcube_pi0 \
#     n_envs=8 num_test_steps=100 proprioception_type=ee_pose \
#     model_cfg.action_horizon=10 model_cfg.model_device=cuda:1
#
# Example 4: Change video FPS and specify model
#   bash eval_openpi.sh stackcube_pi0 \
#     frame_per_second=30 \
#     model_cfg.pytorch_weight_path=/home/tonghe/Project/PretrainedModels/pi05_sft/Pi05-ManiSkill-25Main-SFT
#
# Example 5: Override n_envs and num_test_steps
#   bash eval_openpi.sh spoon_pi05 \
#     n_envs=8 \
#     num_test_steps=50
#
# Example 6: Complete override with all common params
#   bash eval_openpi.sh spoon_pi05 \
#     n_envs=10 \
#     num_test_steps=40 \
#     frame_per_second=24
#
# Example 7: Enable verbose step_infos logging (default logs: instructions, success, rewards, actions, proprioception)
#   bash eval_openpi.sh spoon_pi05 \
#     save_detailed_logs=true
#
# NOTE: Default logs already include instructions, success rates, rewards, actions, and proprioception.
#       Setting save_detailed_logs=true adds verbose step_infos (makes files ~100x larger).
#     
#
# ============================================================================

# Parse arguments
COMPOSED_CONFIG="${1:-stackcube_pi05}"       # Default: stackcube_pi0
shift 1 2>/dev/null || shift $#             # Remove first arg, rest are Hydra overrides

# Environment setup
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Framework settings
export USE_TF=0
export USE_TORCH=1
export TF_CPP_MIN_LOG_LEVEL=3

# Python paths
# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
QUICK_TEST_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
MANIP_DIR="$( cd "$QUICK_TEST_DIR/.." && pwd )"
PROJECT_DIR="$( cd "$MANIP_DIR/.." && pwd )"
LEROBOT_DIR="$PROJECT_DIR/lerobot"
SIMPLERENVPLUS_DIR="$MANIP_DIR/SimplerEnvPlus"

# Set PYTHONPATH with relative-based paths
export PYTHONPATH="$QUICK_TEST_DIR:$MANIP_DIR:$PYTHONPATH"
export PYTHONPATH="$LEROBOT_DIR:$PYTHONPATH"
export PYTHONPATH="$SIMPLERENVPLUS_DIR:$PYTHONPATH"
export PYTHONPATH="/home/tonghe/Project/openpi-main/src:$PYTHONPATH"
export PYTHONPATH="/home/tonghe/Project/openpi-main/packages/openpi-client/src:$PYTHONPATH"
export PATH="/home/tonghe/Project/openpi-main-portable/openpi-main/.venv/bin:$PATH"

# Activate environment
source /home/tonghe/Project/openpi-main-portable/openpi-main/.venv/bin/activate

# Build command - simple and clean!
CMD="python $SCRIPT_DIR/eval_openpi.py \
    --config-dir=$SCRIPT_DIR \
    --config-name=configs/${COMPOSED_CONFIG} \
    frame_per_second=16 \
    n_envs=10 \
    num_test_steps=50 \
    model_cfg.pytorch_weight_path=/home/tonghe/Project/PretrainedModels/pi05_sft/Pi05-ManiSkill-25Main-SFT/ \
    model_cfg.norm_stats_dir=/home/tonghe/Project/PretrainedModels/pi05_sft/Pi05-ManiSkill-25Main-SFT/assets/PutOnPlateInScene25Mainv3/meta \
    $@"


eval $CMD
