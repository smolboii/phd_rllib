from typing import Any, Mapping, Union

import torch
import torch.nn as nn

class ModelWrapper():

    # Base class for wrappers around regular pytorch networks or rllib models that are used throughout this repository.

    def __init__(self, model: nn.Module, device: str):
        super().__init__()

        self.model = model
        self.model.to(device)
        
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward(x.to(self.device))
    
