import torch
from torch import Tensor

def shuffle_tensor(tensor: Tensor):
    shuffled_inds = torch.randperm(tensor.shape[0], device=tensor.device)
    return tensor[shuffled_inds]

def shuffle_tensors(*tensors: Tensor):
    # shuffles all the passed tensors using the same indices (must all have same number of elements and be on same device)

    num_el = tensors[0].shape[0]
    dev = tensors[0].device

    if any([t.shape[0] != num_el for t in tensors]):
        raise Exception("tensors must all have same number of elements")
    if any([t.device != dev for t in tensors]):
        raise Exception("tensors must all be on same device")
        
    shuffled_inds = torch.randperm(num_el, device=dev)
    return (tensor[shuffled_inds] for tensor in tensors)