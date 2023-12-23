import torch
from torch import Tensor

def shuffle_tensor(tensor: Tensor):
    shuffled_inds = torch.randperm(tensor.shape[0], device=tensor.device)
    return tensor[shuffled_inds]

def shuffle_tensors(*tensors: Tensor):
    # shuffles all the passed tensors using the same indices (must all be same shape and on same device)

    t_shape = tensors[0].shape
    dev = tensors[0].device

    if any([t.shape != t_shape for t in tensors]):
        raise Exception("tensors must all be same shape")
    if any([t.device != dev for t in tensors]):
        raise Exception("tensors must all be on same device")
        
    shuffled_inds = torch.randperm(t_shape[0], device=dev)
    return (tensor[shuffled_inds] for tensor in tensors)