import os
import torch
import torch.distributed as dist
from utils.custom_memory_manager import cleanup_cuda_memory

def setup_distributed():
    """Setup distributed training
    Return: three integers, which are the rank, world_size, local_rank
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        torch.distributed.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            device_id=local_rank  # only for pytorch>2.6
        )

        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        # Single GPU or CPU training
        return 0, 1, 0

def cleanup_and_killprocess(world_size):
    cleanup_cuda_memory()
    if world_size > 1:
        dist.destroy_process_group()
        
        
        