import os
import sys
import logging
import json
import math
import random
import time
import numpy as np
import shutil
import json

from datetime import timedelta
from typing import Dict, Any, List

import filtros
filtros.activate()

from qqdm import qqdm
from matplotlib import pyplot as plt
import torch
from torch.nn import MSELoss
import ray
from ray.train.torch import get_device
from ray.rllib.algorithms.dqn.dqn_torch_policy import build_q_model_and_distribution
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.dqn import DQNConfig, DQN
from ray.rllib.algorithms.apex_dqn import ApexDQNConfig, ApexDQN
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.tune.logger import UnifiedLogger
import dacite
import gymnasium as gym

from vae import VAE
from lifelong.core.lifelong_learner import LifelongLearner
from lifelong.config import LifelongLearnerConfig
from lifelong.replay.buffers.generative import GenerativeObservationBuffer, cyclic_anneal_creator
from lifelong.replay.buffers.raw import RawObservationBuffer
from lifelong.plugins.buffers import RawObservationBufferPlugin, GenerativeObservationBufferPlugin
from lifelong.log import init_logger
from lifelong.util import deep_update


if __name__ == "__main__":

    try:
        with open("config.json") as cf_fp:
            cf_dict = json.loads(cf_fp.read())
            cf = dacite.from_dict(LifelongLearnerConfig, cf_dict, config=dacite.Config(strict=True))
    except FileNotFoundError:
        cf = LifelongLearnerConfig()

    log_dir = os.path.join('logs', cf.exp_name)
    if os.path.exists(log_dir):
        raise Exception("Logging directory already exists - either delete this directory or choose a different experiment name")
    os.makedirs(log_dir)

    dummy_logdir = 'dummy_logs'
    try:
        shutil.rmtree(dummy_logdir)
    except FileNotFoundError:
        pass
    os.mkdir(dummy_logdir)

    ray.init(
        _system_config={
            "object_spilling_config": json.dumps(
                {"type": "filesystem", "params": {"directory_path": "/data/jayl164/ray_sessions/"}},
            )
        },
        log_to_driver=False,
    )

    logger = init_logger(log_dir)

    if cf.shuffle_envs:
        random.shuffle(cf.env_names)

    n_replay_shards = 4

    wake_algo_config = ApexDQNConfig()
    replay_config = {
        "capacity": 1_000_000,
        "num_replay_buffer_shards": n_replay_shards
    }
    wake_algo_config = wake_algo_config.training(replay_buffer_config=replay_config,
                            target_network_update_freq=50_000, 
                            num_steps_sampled_before_learning_starts=100_000,
                            train_batch_size=cf.wake_config.batch_size,
                            lr=cf.wake_config.lr)
    wake_algo_config = wake_algo_config.exploration(exploration_config={
        "initial_epsilon": 1,
        "final_epsilon": 0.05,
        "epsilon_timesteps": 10_000,
    })
    wake_algo_config = wake_algo_config.resources(num_gpus=1)
    wake_algo_config = wake_algo_config.rollouts(num_rollout_workers=16, num_envs_per_worker=16)

    env_config = {
        "full_action_space": True,
    }

    os.environ["CUDA_VISIBLE_DEVICES"] = cf.cuda_visible_devices
    device = get_device()

    buffer = None
    buffer_plugin = None
    if cf.buffer_config.type == "raw":
        buffer = RawObservationBuffer((4,84,84), device="cpu")
        buffer_plugin = RawObservationBufferPlugin(buffer)
    
    elif cf.buffer_config.type == "generative":
        gen_model_creator = lambda: VAE(4, 1024, desired_hw=(84,84), kld_beta=cf.buffer_config.kld_beta, 
                                        log_var_clip_val=cf.buffer_config.log_var_clip_val) # clip val wrong maybe?
        kl_fn = cyclic_anneal_creator(cf.buffer_config.n_epochs_per_train*cf.sleep_config.buffer_capacity//cf.buffer_config.train_batch_size, 
                                    6, 0.8, end_beta = 0.0001)
        buffer = GenerativeObservationBuffer(
            gen_model_creator,
            (4,84,84),
            config=cf.buffer_config,
            #kl_beta_annealing_fn=kl_fn,
            log_dir=os.path.join(log_dir, "gen"),
            device=device
        )
        buffer_plugin = GenerativeObservationBufferPlugin(buffer)

    sleep_algo_config = DQNConfig()
    sleep_algo_config = sleep_algo_config.evaluation(evaluation_duration=200, evaluation_duration_unit="episodes", evaluation_interval=1, evaluation_num_workers=16)

    def wake_buffer_collector_fn(apexdqn: ApexDQN, amt):
        replay_mgr = apexdqn._replay_actor_manager
        results = replay_mgr.foreach_actor(
                func=lambda actor: actor.sample(amt//n_replay_shards),
                remote_actor_ids=list(range(n_replay_shards))  # split amount across all shards, with a single batch each to ensure no duplicate experiences
        ).result_or_errors

        return [ r.get().policy_batches['default_policy'] for r in results ]
    
    def wake_learner_algo_instantiator(env_name: str):
        wake_cf = wake_algo_config.copy(copy_frozen=False)
        wake_cf = wake_cf.environment(env=env_name, env_config=env_config)

        logger_creator = lambda cf: UnifiedLogger(cf, logdir=os.path.join(log_dir, env_name))
        return ApexDQN(config=wake_cf.copy(), logger_creator=logger_creator)
    
    def sleep_algo_instantiator(env_name: str, model_config: dict):
        sleep_cf = sleep_algo_config.copy(copy_frozen=False)
        sleep_cf.model.update(model_config)
        sleep_cf = sleep_cf.environment(env=env_name, env_config=env_config)

        return DQN(config=sleep_cf, logger_creator=lambda cf: UnifiedLogger(cf, logdir="dummy_logs"))

    lifelong_learner = LifelongLearner(
        cf.env_names, 
        wake_learner_algo_instantiator,
        sleep_algo_instantiator,
        buffer,
        cf,
        wake_buffer_collector_fn,
        logger=logger,
        sleep_logdir=os.path.join(log_dir, "sleep"),
        device=device
    )

    lifelong_learner.plugins.append(buffer_plugin)
    lifelong_learner.learn()