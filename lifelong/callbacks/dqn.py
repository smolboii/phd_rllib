import os
from typing import Type

import torch.nn as nn
from torch import Tensor
from gymnasium import Space
from ray.rllib.algorithms import Algorithm, AlgorithmConfig
from ray.rllib.algorithms.dqn import DQN, DQNConfig
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.tune.logger import UnifiedLogger

from lifelong.callbacks.base import AlgorithmCallbackWrapper
from lifelong.models.wrappers.base import ModelWrapper
from lifelong.models.wrappers.dqn import DQNModelWrapper

class DQNCallbackWrapper(AlgorithmCallbackWrapper):

    def __init__(self, algo_config: DQNConfig, add_value: bool = False, env_config: dict = {}, model_config: dict = {}):
        super().__init__(DQN, algo_config, env_config, model_config)
        self.add_value = add_value

    def instantiate_algorithm(self, env_name: str, log_dir: str) -> DQN:
        return super().instantiate_algorithm(env_name, log_dir)
    
    def wrap_model(self, model: nn.Module, device: str) -> DQNModelWrapper:
        return DQNModelWrapper(
            model,
            self.add_value,
            device
        )
    
    def get_model(self, algo: DQN) -> DQNModelWrapper:
        return self.wrap_model(algo.get_policy().model)
    
    def instantiate_model(self, env_name: str, device: str) -> DQNModelWrapper:
        return self.wrap_model(self.instantiate_algorithm(env_name, "dummy_logs").get_policy().model, device)