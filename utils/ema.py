
import torch  
import torch.nn as nn


def update_ema_parameters(ema_model: nn.Module, model: nn.Module, alpha: float = 0.995):
    """Update EMA parameters in place with ema_param <- alpha * ema_param + (1 - alpha) * param."""
    for ema_module, module in zip(ema_model.modules(), model.modules(), strict=True):
        for (n_p_ema, p_ema), (n_p, p) in zip(
            ema_module.named_parameters(recurse=False), 
            module.named_parameters(recurse=False), 
            strict=True
        ):
            assert n_p_ema == n_p, "Parameter names don't match for EMA model update"
            if isinstance(p, dict):
                raise RuntimeError("Dict parameter not supported")
            if isinstance(module, nn.modules.batchnorm._BatchNorm) or not p.requires_grad:
                # Copy BatchNorm parameters, and non-trainable parameters directly.
                p_ema.copy_(p.to(dtype=p_ema.dtype).data)
            else:
                with torch.no_grad():
                    p_ema.mul_(alpha)
                    p_ema.add_(p.to(dtype=p_ema.dtype).data, alpha=1 - alpha)
                    
                    
                    
                    