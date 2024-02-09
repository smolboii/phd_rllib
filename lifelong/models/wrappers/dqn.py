from typing import Union

import torch
import torch.nn as nn
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel

from lifelong.models.wrappers.base import ModelWrapper

class DQNModelWrapper(ModelWrapper):

    def __init__(self, model: Union[DQNTorchModel, nn.Module], add_value: bool, device: str):
        super().__init__(model, device)
        self.model: Union[DQNTorchModel, nn.Module]  # update type hint
        self.add_value = add_value

    def forward(self, x: torch.Tensor):
        return DQNModelWrapper.static_forward(x, self.model, self.add_value)
    
    def static_forward(x: torch.Tensor, model:  Union[DQNTorchModel, nn.Module], add_value: bool):
        features, _ = model.forward({"obs": x}, None, None)
        return_val = model.advantage_module(features)

        if add_value:  # whether we should add the observation value to the individual action advantages
            return_val += model.value_module(features)

        return return_val
    
class DQNFeatureModelWrapper(ModelWrapper):

    # Alternative wrapper that only outputs the features (used for dual network plugin)

    def __init__(self, model: Union[DQNTorchModel, nn.Module], device: str):
        super().__init__(model, device)
        self.model: Union[DQNTorchModel, nn.Module]  # update type hint

    def forward(self, x: torch.Tensor):
        features, _ = self.model.forward({"obs": x}, None, None)
        return features