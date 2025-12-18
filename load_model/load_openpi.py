"""
Simple script to load a pre-trained/SFT/RLFT pi0/pi05 model onto a single GPU. 
"""



import dataclasses
import gc
import logging
import os
import pathlib

import numpy as np
import safetensors.torch
import torch

import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.training.config as _config


def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def get_model_state_dict(model):
    """Get state dict from model, handling DDP wrapper."""
    return (
        model.module.state_dict()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.state_dict()
    )


def get_model_parameters(model):
    """Get parameters from model, handling DDP wrapper."""
    return (
        model.module.parameters()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.parameters()
    )


def load_weights_lenient(model, safetensors_path: pathlib.Path, device: torch.device) -> None:
    """Load weights from a safetensors file, ignoring unexpected keys.

    - Filters out parameters not present in the current model
    - Handles optional 'module.' prefix from DDP checkpoints
    - Applies common key normalizations (best-effort) and uses strict=False
    """
    if not safetensors_path.exists():
        raise FileNotFoundError(f"No model checkpoint found at {safetensors_path}")

    # Choose correct model object (unwrap DDP)
    model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

    # Current model keys
    model_state = model_to_load.state_dict()
    model_keys = set(model_state.keys())

    # Load raw tensors from safetensors
    raw_state = safetensors.torch.load_file(str(safetensors_path), device=str(device))

    def normalize_key(k: str) -> str:
        # Strip leading 'module.' if present (from DDP checkpoints)
        if k.startswith("module."):
            k = k[len("module."):]
        
        # Remove '.model.' from paligemma paths - checkpoint has .model. but model doesn't
        # The checkpoint structure is: paligemma_with_expert.paligemma.model.language_model.*
        # The model structure is:      paligemma_with_expert.paligemma.language_model.*
        # So we need to remove the .model. part
        if ".model.language_model" in k:
            k = k.replace(".model.language_model", ".language_model")
        if ".model.vision_tower" in k:
            k = k.replace(".model.vision_tower", ".vision_tower")
        
        # More general: remove any .model. that appears after paligemma
        # This handles edge cases where .model. appears in other contexts
        # Split by '.' and remove 'model' when it appears right after 'paligemma'
        parts = k.split(".")
        new_parts = []
        i = 0
        while i < len(parts):
            # If we see "paligemma" followed by "model", skip the "model" part
            if i + 1 < len(parts) and parts[i] == "paligemma" and parts[i + 1] == "model":
                new_parts.append(parts[i])  # Add "paligemma"
                i += 2  # Skip both "paligemma" and "model"
            else:
                new_parts.append(parts[i])
                i += 1
        k = ".".join(new_parts)
        
        return k

    # Build filtered state dict - try multiple normalization strategies
    filtered_state: dict[str, torch.Tensor] = {}
    unexpected_keys: list[str] = []
    matched = 0
    
    for k, v in raw_state.items():
        # Try normalized key first
        nk = normalize_key(k)
        
        # If normalized key matches, use it
        if nk in model_keys:
            filtered_state[nk] = v
            matched += 1
        # Also try original key in case model has it
        elif k in model_keys:
            filtered_state[k] = v
            matched += 1
        else:
            unexpected_keys.append(k)

    # Report
    missing_after = [k for k in model_keys if k not in filtered_state]
    logging.info(
        f"Lenient weight load: matched={matched}/{len(model_keys)} model keys, "
        f"unexpected={len(unexpected_keys)} checkpoint keys, missing={len(missing_after)} model keys"
    )
    
    # Show sample of matched keys to verify normalization worked
    if matched > 0:
        matched_samples = [k for k in filtered_state.keys()][:3]
        logging.info(f"Matched keys (top 3 samples): {', '.join(matched_samples)}")
    
    if len(unexpected_keys) > 0:
        # Show first few unexpected keys to help debug
        sample = ", ".join(unexpected_keys)
        logging.info(f"Ignored unexpected keys, total number is {len(unexpected_keys)}(all of them are shown): {sample}")
    
    if len(missing_after) > 0:
        # Show first few missing keys to help debug
        missing_sample = ", ".join(missing_after)
        logging.warning(f"Missing model keys not in checkpoint, total number is {len(missing_after)}(all of them are shown): {missing_sample}")

    # Finally, load with strict=False so any non-critical missing keys are tolerated
    missing, unexpected = model_to_load.load_state_dict(filtered_state, strict=False)
    # 'unexpected' should be empty because we filtered; keep this log for safety
    if missing:
        logging.warning(f"Final missing keys after load_state_dict (sample): {', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''}")
    if unexpected:
        logging.warning(f"Unexpected keys after filtering (should be 0): {unexpected[:5]}")


def load_checkpoint(model, optimizer, checkpoint_dir, device):
    """Load the latest checkpoint and return the global step."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]

    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"

    # Clear memory before loading checkpoints
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        # Load model state with error handling
        logging.info("Loading model state...")
        safetensors_path = ckpt_dir / "model.safetensors"

        if safetensors_path.exists():
            model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            safetensors.torch.load_model(model_to_load, safetensors_path, device=str(device))
            logging.info("Loaded model state from safetensors format")
        else:
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_model")

        # Load optimizer state with error handling
        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"

        if optimizer_path.exists():
            optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
            logging.info("Loaded optimizer state from pt format")
        else:
            raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_optimizer")

        # Load metadata
        logging.info("Loading metadata...")
        metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
        global_step = metadata.get("global_step", latest_step)
        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_metadata")

        logging.info(f"Successfully loaded all checkpoint components from step {latest_step}")
        return global_step

    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clear memory and provide detailed error message
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            log_memory_usage(device, latest_step, "after_oom_error")
            raise RuntimeError(
                "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ) from e
        raise

def log_memory_usage(device, step, phase="unknown"):
    """Log detailed memory usage information."""
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    memory_free = memory_free / 1e9

    # Get more detailed memory info
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB"
    )


def load_openpi_model(model_config: _config.TrainConfig, model_device: torch.device):
    set_seed(model_config.seed, 0)
    device = model_device
    # Build model
    if not isinstance(model_config.model, openpi.models.pi0_config.Pi0Config):
        # Convert dataclass to Pi0Config if needed
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=model_config.pytorch_training_precision,
            action_dim=model_config.model.action_dim,
            action_horizon=model_config.model.action_horizon,
            max_token_len=model_config.model.max_token_len,
            paligemma_variant=getattr(model_config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(model_config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(model_config.model, "pi05", False),
        )
    else:
        model_cfg = model_config.model
        # Update dtype to match pytorch_training_precision
        object.__setattr__(model_cfg, "dtype", model_config.pytorch_training_precision)

    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)

    if hasattr(model, "gradient_checkpointing_enable"):
        enable_gradient_checkpointing = True
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for memory optimization")
    else:
        enable_gradient_checkpointing = False
        logging.info("Gradient checkpointing is not supported for this model")

    # Log initial memory usage after model creation
    if torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # Load weights from weight_loader if specified (for fine-tuning)
    if model_config.pytorch_weight_path is not None:
        logging.info(f"Loading weights from: {model_config.pytorch_weight_path}")

        model_path = os.path.join(model_config.pytorch_weight_path, "model.safetensors")

        # Load weights from SFT ckpt.
        safetensors.torch.load_model(
            model, model_path
        )
        logging.info(f"Loaded Pre-trained/SFT PyTorch weights from {model_config.pytorch_weight_path}")

        # # Load weights from RLFT ckpt.
        # load_weights_lenient(model, pathlib.Path(model_path), device)
        # logging.info(f"Loaded RLFT PyTorch weights from {config.pytorch_weight_path}")

    model.eval()

    return model

    
def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="pi0_libero") # pi05_libero
    parser.add_argument("--mode", type=str, default="few") # few
    parser.add_argument("--tasks", type=str, default="all") # all, libero_10
    parser.add_argument("--exp_name", type=str, default="test") # train expert only
    parser.add_argument("--checkpoint_suffix", type=str, default=None, help="Optional suffix to append to exp_name in checkpoint directory path")
    
    # Add these lines:
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--peak_lr", type=float, default=None)
    parser.add_argument("--decay_steps", type=int, default=None)
    parser.add_argument("--decay_lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--discrete_state_input", type=bool, default=None) # type: ignore
    parser.add_argument("--action_horizon", type=int, default=None)
    parser.add_argument("--action_chunk", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--save_interval", type=int, default=None)
    parser.add_argument("--num_train_steps", type=int, default=None)
    parser.add_argument("--skip_state_embedding", type=bool, default=None) # type: ignore
    parser.add_argument("--pytorch_weight_path", type=str, default=None)
    parser.add_argument("--model_device", type=str, default="cuda:0")
    return parser.parse_args()

def load_openpi_model_config(args):
    from openpi.training import config as _config
    logging.info(f"Config name: {args.config_name}")
    config_name = args.config_name
    mode = args.mode
    # Construct exp_name with optional suffix
    exp_name = args.exp_name
    if args.checkpoint_suffix is not None:
        exp_name = f"{exp_name}{args.checkpoint_suffix}"
    # config
    config = _config.get_config(config_name)

    # override model config from prompt
    if args.batch_size is not None:
        config = dataclasses.replace(config, batch_size=args.batch_size)
    if args.discrete_state_input is not None:
        config = dataclasses.replace(config, model=dataclasses.replace(config.model, discrete_state_input=args.discrete_state_input))
    if args.skip_state_embedding is not None:
        config = dataclasses.replace(config, model=dataclasses.replace(config.model, skip_state_embedding=args.skip_state_embedding))
    # override model config from prompt
    if args.action_horizon is not None or args.action_chunk is not None:
        model_kwargs = {}
        if args.action_horizon is not None:
            model_kwargs['action_horizon'] = args.action_horizon
        if args.action_chunk is not None:
            model_kwargs['action_chunk'] = args.action_chunk
        if args.num_steps is not None:
            model_kwargs['num_steps'] = args.num_steps
        if args.save_interval is not None:
            config=dataclasses.replace(config, save_interval=args.save_interval)
        if args.num_train_steps is not None:
            config=dataclasses.replace(config, num_train_steps=args.num_train_steps)
        
        new_model = dataclasses.replace(config.model, **model_kwargs)
        config = dataclasses.replace(config, model=new_model)
    # full dataset libero 
    if mode == "libero-full":
        config = dataclasses.replace(config, exp_name=args.exp_name)
    # few shot libero
    elif mode == "libero-few":
        # single task training
        if args.tasks == "libero_object":
            new_base_config = dataclasses.replace(
                config.data.base_config, 
                episodes=[
                    807, 808, 810, 811, 813, 814, 816, 819, 821, 823, # object
                ]
            )
        elif args.tasks == "libero_spatial":
            new_base_config = dataclasses.replace(
                config.data.base_config, 
                episodes=[
                    1261, 1262, 1264, 1267, 1272, 1278, 1280, 1281, 1283, 1290 # spatial
                ]
            )
        elif args.tasks == "libero_goal":
            new_base_config = dataclasses.replace(
                config.data.base_config, 
                episodes=[
                    379, 380, 382, 384, 385, 386, 388, 392, 398, 399, # goal 
                ]
            )
        # TODO: LIBERO-10
        # libero-10 3-shot
        elif args.tasks == "libero_10":
            new_base_config = dataclasses.replace(
                config.data.base_config, 
                episodes=[
                    0, 18, 22,1, 4, 5,2, 3, 34,6, 38, 40,7, 9, 25,8, 13, 26,10, 20, 23,12, 17, 24,14, 15, 16,27, 28, 47
                ]
            )
        else:
            raise ValueError(f"Invalid task: {args.tasks}")
        # replace the data config
        config = dataclasses.replace(
            config,
            data=dataclasses.replace(config.data, base_config=new_base_config)
        )
        # other configs
        config = dataclasses.replace(config, exp_name=args.exp_name)
        config = dataclasses.replace(config,num_train_steps=30_000)
    elif mode=='maniskill':
        # Parse task name from --tasks formatted as "maniskill_<TASK>"
        task_name = args.tasks.split("_", 1)[1] if "_" in args.tasks else args.tasks
        # Set dataset root to the specific ManiSkill task directory
        new_base_config = dataclasses.replace(
            config.data.base_config,
            root=f"./data/maniskill/{task_name}",
        )
        # Override assets_dir and repo_id for ManiSkill
        new_assets_config = dataclasses.replace(
            config.data.assets,
            assets_dir=f"./data/maniskill/"
        )
        # replace the data config
        config = dataclasses.replace(
            config,
            data=dataclasses.replace(
                config.data, 
                base_config=new_base_config,
                assets=new_assets_config,
                repo_id=f"{task_name}/meta"
            )
        )
        # other configs
        config = dataclasses.replace(config, exp_name=exp_name)
    
    # Optional override for weights path, restricted to pi0/pi05 maniskill configs
    if args.pytorch_weight_path is not None:
        try:
            config_name_current = config.name  # may be suppressed in types, but present at runtime
        except Exception:
            config_name_current = None
        if config_name_current in ("pi0_maniskill", "pi05_maniskill"):
            config = dataclasses.replace(config, pytorch_weight_path=args.pytorch_weight_path)

    config = dataclasses.replace(config, checkpoint_base_dir=f"./checkpoints/sft/")
    
    return config

if __name__ == "__main__":
    args = parse_args()
    
    model_config = load_openpi_model_config(args)
    print(f"model_config: {model_config}")
    
    model_device = args.model_device
    model = load_openpi_model(model_config, model_device)
    print(f"model: {model}")