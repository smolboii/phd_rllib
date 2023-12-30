import os
from typing import Type
from ray.rllib.algorithms import Algorithm, AlgorithmConfig

from ray.rllib.algorithms.dqn import DQN, DQNConfig
from ray.tune.logger import UnifiedLogger
from torch import Tensor

from lifelong.callbacks.base import AlgorithmCallbackWrapper

class DQNCallbackWrapper(AlgorithmCallbackWrapper):

    def __init__(self, algo_config: DQNConfig, env_config: dict = {}, model_config: dict = {}):
        super().__init__(DQN, algo_config, env_config, model_config)

    def instantiator(self, env_name: str, log_dir: str) -> DQN:
        return super().instantiator(env_name, log_dir)
    
    def buffer_collector(self, algo: Algorithm, amt: int) -> Tensor:
        return super().buffer_collector(algo, amt)