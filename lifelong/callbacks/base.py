from typing import TypeVar, Type

from torch import Tensor
from ray.tune.logger import UnifiedLogger
from ray.rllib.algorithms import Algorithm, AlgorithmConfig

class AlgorithmCallbackWrapper:
    def __init__(self, algo_type: Type[Algorithm], algo_config: AlgorithmConfig, env_config: dict = {}, model_config: dict = {}):
        self.algo_type = algo_type
        self.algo_config = algo_config
        self.env_config = env_config
        self.model_config = model_config

    def instantiator(self, env_name: str, log_dir: str) -> Algorithm:
        config = self.algo_config.copy(copy_frozen=False)
        config.model.update(self.model_config)
        config = config.environment(env=env_name, env_config=self.env_config)

        return self.algo_type(config=config.copy(), logger_creator=lambda cf: UnifiedLogger(cf, logdir=log_dir))
    
    def buffer_collector(self, algo: Algorithm, amt: int) -> Tensor:
        raise NotImplementedError()