#!/bin/bash

# Run test_env_wrapper_overlay.py with Hydra outputs redirected to log directory

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
QUICK_TEST_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
MANIP_DIR="$( cd "$QUICK_TEST_DIR/.." && pwd )"
PROJECT_DIR="$( cd "$MANIP_DIR/.." && pwd )"
LEROBOT_DIR="$PROJECT_DIR/lerobot"
SIMPLERENVPLUS_DIR="$MANIP_DIR/SimplerEnvPlus"

# Set PYTHONPATH with relative-based paths
export PYTHONPATH="$QUICK_TEST_DIR:$MANIP_DIR:$PYTHONPATH"
export PYTHONPATH="$LEROBOT_DIR:$PYTHONPATH"
export PYTHONPATH="$SIMPLERENVPLUS_DIR:$PYTHONPATH"

# Default config
CONFIG_NAME="${1:-positionchangeto-test}"

# Define Hydra output directory relative to the log directory
LOG_BASE_DIR="$SCRIPT_DIR/../log/quick_test_maniskill/hydra_outputs"

# Run the test script with Hydra output redirected
python "$SCRIPT_DIR/test_env_wrapper_overlay.py" \
    --config-dir="$SCRIPT_DIR/env_configs/simpler_messy_pp" \
    --config-name="$CONFIG_NAME" \
    hydra.run.dir="$LOG_BASE_DIR/\${hydra.job.name}/\${now:%Y-%m-%d_%H-%M-%S}"

