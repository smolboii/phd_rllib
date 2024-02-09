from typing import Type

from torch import Tensor
import torch.nn as nn
from ray.tune.logger import UnifiedLogger
from ray.rllib.algorithms import Algorithm, AlgorithmConfig

from lifelong.models.wrappers.base import ModelWrapper

class AlgorithmCallbackWrapper:
    def __init__(self, algo_type: Type[Algorithm], algo_config: AlgorithmConfig, env_config: dict = {}, model_config: dict = {}):
        self.algo_type = algo_type
        self.algo_config = algo_config
        self.env_config = env_config
        self.model_config = model_config

    def instantiate_algorithm(self, env_name: str, log_dir: str) -> Algorithm:
        config = self.algo_config.copy(copy_frozen=False)
        config.model.update(self.model_config)
        config = config.environment(env=env_name, env_config=self.env_config)

        return self.algo_type(config=config, logger_creator=lambda cf: UnifiedLogger(cf, logdir=log_dir))
    
    def wrap_model(self, model: nn.Module, device: str) -> ModelWrapper:
        raise NotImplementedError()

    def get_model(self, algo: Algorithm) -> ModelWrapper:
        raise NotImplementedError()
    
    def instantiate_model(self, env_name: str, device: str) -> ModelWrapper:
        raise NotImplementedError()

    def buffer_collector(self, algo: Algorithm, amt: int) -> Tensor:
        raise NotImplementedError()