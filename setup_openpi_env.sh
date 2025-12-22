#!/bin/bash
# Setup script for openpi environment with all dependencies
# Run this from the openpi-main directory

set -e  # Exit on error

echo "=================================================="
echo "Setting up openpi environment..."
echo "=================================================="


# # Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH. Add this if openpi environment does not detect uv. 
export PATH="$HOME/.local/bin:$PATH"


# Navigate to openpi-main directory
cd ~/Project/openpi-main


echo ""
echo "Step 1: Running uv sync to set up virtual environment..."
GIT_LFS_SKIP_SMUDGE=1 uv sync

echo ""
echo "Step 2: Installing openpi in editable mode..."
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

echo ""
echo "Step 3: Installing transformers 4.53.2..."
source .venv/bin/activate
uv pip install transformers==4.53.2

echo ""
echo "Step 4: Copying transformers_replace files..."
# Dynamically detect Python version in venv
PYTHON_VERSION=$(python -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/$PYTHON_VERSION/site-packages/transformers/

echo ""
echo "Step 5: Installing core ML frameworks (PyTorch, JAX, Flax)..."
uv pip install torch==2.9.0 torchvision==0.24.0 jax==0.6.2 jaxlib==0.6.2 flax==0.10.7

echo ""
echo "Step 6: Installing additional ML dependencies..."
uv pip install augmax beartype "jaxtyping==0.2.36" pytest ml_collections numpydantic datasets jsonlines draccus concurrent_log_handler av wandb sentencepiece tqdm_loggable

echo ""
echo "Step 7: Installing ManiSkill and robotics libraries..."
uv pip install gymnasium mani_skill sapien opencv-python trimesh

echo ""
echo "Step 8: Verifying installation..."
python -c 'from openpi.models_pytorch.pi0_pytorch import PI0Pytorch; print("✓ PI0Pytorch imported successfully!")'
echo ""
echo "=================================================="
echo "✓ Setup complete!"
echo "=================================================="
echo ""
echo "To use this environment, run:"
echo "  cd ~/Project/openpi-main"
echo "  source .venv/bin/activate"
echo ""


