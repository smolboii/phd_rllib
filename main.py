import os
import json
import math
import random
import time
import numpy as np
import shutil

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
import gymnasium as gym

from vae import VAE
from lifelong.core.lifelong_learner import LifelongLearner
from lifelong.core.config import SleepConfig, WakeConfig
from lifelong.replay.buffers.generative import GenerativeObservationBuffer, cyclic_anneal_creator
from lifelong.replay.buffers.raw import RawObservationBuffer
from lifelong.plugins.buffers import RawObservationBufferPlugin, GenerativeObservationBufferPlugin


if __name__ == "__main__":

    exp_name = "temp"
    log_dir = os.path.join('logs', exp_name)
    try:
        shutil.rmtree(log_dir)
    except FileNotFoundError:
        pass  # logging directory doesnt exist anyway
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

    n_replay_shards = 4
    train_batch_size = 256
    lr = 0.00025 / 4
    shuffle_envs = False

    env_names = [
        "ALE/Asteroids-v5",
        # "ALE/Boxing-v5",  # gives negative episode rewards, so kind of annoying for graphing results
        "ALE/SpaceInvaders-v5",
        "ALE/Asterix-v5",
        "ALE/Alien-v5",
        "ALE/Breakout-v5"
    ]
    if shuffle_envs:
        random.shuffle(env_names)

    wake_algo_config = ApexDQNConfig()
    # wake_algo_config["model"]["conv_filters"] = [
    #     [16, [8, 8], 4],
    #     [32, [4, 4], 2],
    #     [64, [3, 3], 2],
    #     [128, [3, 3], 2],
    #     [256, [3, 3], 2],
    #     [256, [2, 2], 2],
    #     [256, [1, 1], 1],
    # ]
    replay_config = {
        "capacity": 1_000_000,
        "type": "MultiAgentPrioritizedReplayBuffer",
        "replay_buffer_shards_colocated_with_driver": True,
        "worker_side_prioritization": True,
        "num_replay_buffer_shards": n_replay_shards
    }
    wake_algo_config = wake_algo_config.training(replay_buffer_config=replay_config,
                            target_network_update_freq=50_000, 
                            num_steps_sampled_before_learning_starts=100_000,
                            train_batch_size=train_batch_size,
                            lr=lr)
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
    
    # new stuff to add:
    # BIG: pass wake buffer observations through generative model before using them for knowledge distillation. otherwise generated examples and wake examples will
    # look even more dissimilar than they are already (as a result of being from different environments), harming performance. the wake examples will still need to be labelled prior to
    # being passed through the generative model, as the wake model expects them to look as they currently are.
    # 1. train generative model jointly each sleep phase on ground truth examples (to test upper bound for generative replay performance)
    # 2. partially resetting instead of hard resetting (regularising towards standard normal e.g.?)
    # 3. try larger sleep model (could be reaching limit of representational capacity)
    # 4. reset head of sleep network / increase plasiticty of head of sleep network at start of each sleep phase, as feature extractor should be more generalisable than the head
    # of the policy network, making it more suited for just being copied over.

    # thoughts:
    # a. Could copy feature extractor weights from sleep to wake model, and regularise wake model's feature extractor to stay reasonably close to the sleep model's feature extractor.
    # This could then make the knowledge distillation conflict less with the pseudo-rehearsal distillation, as the outputs being distilled will be more similar in nature? (especially
    # if we try distilling at the feature level, e.g.)

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    device = get_device()
    
    ll_sleep_conf = SleepConfig(sleep_epochs=1, eval_at_start_of_sleep=False, reinit_sleep_model=False, 
                                copy_first_task_weights=True, model_config=dict())
    ll_wake_conf = WakeConfig(timesteps_per_env=20_000_000, model_config=dict())

    gen_model_creator = lambda: VAE(4, 1024, desired_hw=(84,84), kld_beta=0.0001, log_var_clip_val=5) # clip val wrong maybe?
    n_gen_epochs = 50
    gen_batch_size = 256
    kl_fn = cyclic_anneal_creator(n_gen_epochs*ll_sleep_conf.buffer_capacity//gen_batch_size, 6, 0.8, end_beta = 0.0001)
    gen_obs_buffer = GenerativeObservationBuffer(
        gen_model_creator,
        (4,84,84),
        n_gen_epochs,
        256,
        #kl_beta_annealing_fn=kl_fn,
        log_interval=10,
        log_dir=os.path.join(log_dir, "gen"),
        reset_model_before_train=True,
        gen_obs_batch_prop=None,
        raw_buffer_capacity_per_task=1_000,
        raw_buffer_obs_batch_prop=0.33,
        device=device
    )

    raw_obs_buffer = RawObservationBuffer((4,84,84), device="cpu")

    sleep_algo_config = DQNConfig()
    sleep_algo_config = sleep_algo_config.evaluation(evaluation_duration=200, evaluation_duration_unit="episodes", evaluation_interval=1, evaluation_num_workers=16)

    larger_sleep_model_config = {
        "conv_filters": [
            [16, [8, 8], 4],
            [32, [4, 4], 2],
            [64, [3, 3], 2],
            [128, [3, 3], 2],
            [256, [3, 3], 2],
            [256, [2, 2], 2],
            [256, [1, 1], 1],
        ]
    }

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
        env_names, 
        wake_learner_algo_instantiator,
        sleep_algo_instantiator,
        gen_obs_buffer, #raw_obs_buffer,
        wake_buffer_collector_fn,
        wake_config=ll_wake_conf,
        sleep_config=ll_sleep_conf,
        sleep_logdir=os.path.join(log_dir, "sleep"),
        device=device
    )
    lifelong_learner.plugins.append(GenerativeObservationBufferPlugin(gen_obs_buffer))
    #lifelong_learner.plugins.append(RawObservationBufferPlugin(raw_obs_buffer))
    lifelong_learner.learn()