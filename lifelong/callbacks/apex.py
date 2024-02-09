import os

import torch
from torch import Tensor
from gymnasium import Space
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.algorithms.apex_dqn import ApexDQN, ApexDQNConfig

from lifelong.callbacks.base import AlgorithmCallbackWrapper
from lifelong.callbacks.dqn import DQNCallbackWrapper
from lifelong.models.wrappers.dqn import DQNModelWrapper

class ApexCallbackWrapper(DQNCallbackWrapper):

    def __init__(self, algo_config: ApexDQNConfig, n_replay_shards: int = 4, add_value: bool = False, env_config: dict = {}, model_config: dict = {}):
        super().__init__(algo_config, add_value, env_config, model_config)
        self.algo_type = ApexDQN  # bit of a hack to set it manually here, should change
        self.n_replay_shards = n_replay_shards

    def instantiate_algorithm(self, env_name: str, log_dir: str) -> ApexDQN:
        return super().instantiate_algorithm(env_name, log_dir)
    
    def buffer_collector(self, algo: ApexDQN, amt: int) -> Tensor:
        replay_mgr = algo._replay_actor_manager
        results = replay_mgr.foreach_actor(
                func=lambda actor: actor.sample(amt//self.n_replay_shards),
                remote_actor_ids=list(range(self.n_replay_shards))  # split amount across all shards, with a single batch each to ensure no duplicate experiences
        ).result_or_errors

        sample_batches = [ r.get().policy_batches['default_policy'] for r in results ]
        obs_batches = []
        for sample_batch in sample_batches:
            obs = torch.from_numpy(sample_batch[sample_batch.OBS])
            obs_batches.append(obs)

        del sample_batches
        # reorganize dimensions so the channel dimension is second ([NxCxHxW]) as opposed to last ([NxHxWxC])
        obs_tensor = torch.unsqueeze(torch.cat(obs_batches, dim=0), 1)
        obs_tensor = torch.squeeze(torch.swapdims(obs_tensor, 1, 4)).to("cpu")

        return obs_tensor