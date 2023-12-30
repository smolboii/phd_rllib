import os
import json
import random
import json
import argparse
import dataclasses

import filtros
filtros.activate()

import ray
from ray.train.torch import get_device
from ray.rllib.algorithms.dqn.dqn_torch_policy import build_q_model_and_distribution
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.apex_dqn import ApexDQNConfig
import dacite

from vae import VAE
from lifelong.core.lifelong_learner import LifelongLearner
from lifelong.config import LifelongLearnerConfig, parse_config_json
from lifelong.replay.buffers.generative import GenerativeObservationBuffer, cyclic_anneal_creator
from lifelong.replay.buffers.raw import RawObservationBuffer
from lifelong.plugins.buffers import RawObservationBufferPlugin, GenerativeObservationBufferPlugin
from lifelong.log import init_logger
from lifelong.callbacks.apex import ApexCallbackWrapper
from lifelong.callbacks.dqn import DQNCallbackWrapper

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config")
    args = arg_parser.parse_args()

    ## custom config args go here, or can parse the passed config file using parse_config_json

    cf = LifelongLearnerConfig()
    cf.exp_name = "test"
    cf.sleep.distillation_type = "kld"
    cf.cuda_visible_devices = "4"

    cf.wake.timesteps_per_env = 100_000
    cf.wake.chkpt_dir = None

    cf.sleep.sleep_epochs = 1
    cf.sleep.eval_at_start_of_sleep = False

    ##

    log_dir = os.path.join('logs', cf.exp_name)
    if os.path.exists(log_dir):
        raise Exception("Logging directory already exists - either delete this directory or choose a different experiment name")
    os.makedirs(log_dir)

    # save a copy of config in this experiments log directory for future reference
    with open(os.path.join(log_dir, "config.json"), "w") as json_fp:
        json.dump(dataclasses.asdict(cf), json_fp, indent=4)

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
                            train_batch_size=cf.wake.batch_size,
                            lr=cf.wake.lr)
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
    if cf.buffer.type == "raw":
        buffer = RawObservationBuffer((4,84,84), cf.buffer, device="cpu")
        buffer_plugin = RawObservationBufferPlugin(buffer)
    
    elif cf.buffer.type == "generative":
        gen_model_creator = lambda: VAE(4, cf.buffer.latent_dims, desired_hw=(84,84), kld_beta=cf.buffer.kld_beta, 
                                        log_var_clip_val=cf.buffer.log_var_clip_val) # clip val wrong maybe?
        kl_fn = cyclic_anneal_creator(cf.buffer.n_epochs_per_train*cf.sleep.buffer_capacity//cf.buffer.train_batch_size, 
                                    6, 0.8, end_beta = 0.0001)
        buffer = GenerativeObservationBuffer(
            gen_model_creator,
            (4,84,84),
            config=cf.buffer,
            #kl_beta_annealing_fn=kl_fn,
            log_dir=os.path.join(log_dir, "gen"),
            device=device
        )
        buffer_plugin = GenerativeObservationBufferPlugin(buffer)

    sleep_algo_config = DQNConfig()
    sleep_algo_config = sleep_algo_config.evaluation(evaluation_duration=200, evaluation_duration_unit="episodes", 
                                                     evaluation_interval=1, evaluation_num_workers=16)

    lifelong_learner = LifelongLearner(
        cf,
        ApexCallbackWrapper(wake_algo_config, n_replay_shards=n_replay_shards, env_config=env_config, model_config=cf.wake.model_config),
        DQNCallbackWrapper(sleep_algo_config, env_config=env_config, model_config=cf.sleep.model_config),
        buffer,
        logger=logger,
        log_dir=log_dir,
        device=device
    )

    lifelong_learner.plugins.append(buffer_plugin)
    lifelong_learner.learn()