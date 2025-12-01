#!/bin/bash

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

# Default config is pusht, but can be overridden by first argument
CONFIG_NAME="${1:-pusht}"

# Run the test script
python "$SCRIPT_DIR/test_env_wrapper_overlay.py" --config-dir="$SCRIPT_DIR/env_configs/simpler_messy_pp" --config-name="$CONFIG_NAME"