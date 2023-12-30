import torch
from torch import softmax


def kld_distillation_loss(outputs: torch.Tensor, targets: torch.Tensor, temperature: float = 1, reduction: str = 'mean'):

    assert len(targets.shape) == len(outputs.shape) == 2

    # add a small eps to log expression to prevent nan
    eps = 1e-7

    sm_targets = softmax(targets / temperature, dim=1)
    sm_outputs = softmax(outputs, dim=1)
    kld_loss = sm_targets * torch.log(sm_targets / sm_outputs + eps)

    if reduction == 'mean':
        return kld_loss.mean()
    
    elif reduction == 'batchmean':
        return kld_loss.sum() / outputs.size(0)
    
    elif reduction == 'sum':
        return kld_loss.sum()
    
    elif reduction == 'none':
        return kld_loss
    
    else:
        raise ValueError(f"Invalid reduction mode specified: '{reduction}'. Must be 'mean', 'batchmean', 'sum', or 'none'.")


def kld_distillation_loss_creator(temperature: float = 1, reduction: str = 'mean'):
    
    def loss_fn(outputs: torch.Tensor, targets: torch.Tensor):
        return kld_distillation_loss(outputs, targets, temperature=temperature, reduction=reduction)
    
    return loss_fn

