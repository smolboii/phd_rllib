import torch
from torch import softmax

def kld_distillation_loss(outputs: torch.Tensor, targets: torch.Tensor, temperature: float = 1):

    assert len(targets.shape) == len(outputs.shape) == 2

    sm_targets = softmax(targets / temperature, dim=1)
    sm_outputs = softmax(outputs, dim=1)

    return torch.mean(sm_targets * torch.log(sm_targets / sm_outputs), dim=1)

def kld_distillation_loss_creator(temperature: float = 1):
    
    def loss_fn(outputs: torch.Tensor, targets: torch.Tensor):
        return kld_distillation_loss(outputs, targets, temperature=temperature)
    
    return loss_fn