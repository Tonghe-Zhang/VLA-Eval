#!/bin/bash

# Run test_env_wrapper_overlay.py with automatic config folder detection

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

# Default config
CONFIG_ARG="${1:-simpler_messy_pp/positionchangeto-test}"

# Define Hydra output directory relative to the log directory
LOG_BASE_DIR="$SCRIPT_DIR/../log/quick_test_maniskill/hydra_outputs"

python "$SCRIPT_DIR/eval_base.py" \
    --config-dir="$SCRIPT_DIR/env_cfg" \
    --config-name="$CONFIG_ARG" \
    hydra.run.dir="$LOG_BASE_DIR/\${hydra.job.name}/\${now:%Y-%m-%d_%H-%M-%S}"

# Usage examples:
# bash ./test_env_wrapper_overlay_script.sh assembly/peginsertion
# bash ./test_env_wrapper_overlay_script.sh assembly/plugcharger
# bash ./test_env_wrapper_overlay_script.sh assembly/assemblingkits
# bash ./test_env_wrapper_overlay_script.sh stacking/stackcube
# bash ./test_env_wrapper_overlay_script.sh stacking/stackpyramid
# bash ./test_env_wrapper_overlay_script.sh drawing/drawsvg
# bash ./test_env_wrapper_overlay_script.sh drawing/drawtriangle
# bash ./test_env_wrapper_overlay_script.sh nonprehensive/rollball
# bash ./test_env_wrapper_overlay_script.sh nonprehensive/pusht
# bash ./test_env_wrapper_overlay_script.sh simpler_messy_pp/position-test
# bash ./test_env_wrapper_overlay_script.sh simpler_messy_pp/carrot
