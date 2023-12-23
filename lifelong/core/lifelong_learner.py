import os
import copy
import tempfile
import datetime
import json
import pickle
import math
import random
import warnings
import time
import numpy as np
import shutil

from datetime import timedelta
from typing import Dict, Any, List, Callable

import filtros
filtros.activate()

import tensorflow as tf
from qqdm import qqdm
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch
from torch import Tensor
from torch.nn import MSELoss
import ray
import logging
from torch.utils.tensorboard import SummaryWriter
from ray.train.torch import get_device
from ray.rllib.models.torch.visionnet import VisionNetwork
from ray.rllib.policy.policy import Policy
from ray.rllib.env.env_context import EnvContext
from ray.rllib.algorithms.dqn.dqn_torch_policy import build_q_model_and_distribution
from ray.rllib.policy.sample_batch import SampleBatch 
from ray.rllib.utils.replay_buffers.multi_agent_prioritized_replay_buffer import MultiAgentPrioritizedReplayBuffer
from ray import tune, air
from ray.tune.stopper.function_stopper import FunctionStopper
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig, DQN, DQNTorchPolicy
from ray.rllib.algorithms.apex_dqn import ApexDQNConfig, ApexDQN
from ray.tune.logger import pretty_print
from ray.tune import register_env
from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind
from ray.rllib.utils.replay_buffers import PrioritizedReplayBuffer, ReplayBuffer
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.tune.logger import UnifiedLogger
import gymnasium as gym

from lifelong.replay.buffers.base import ObservationBuffer
from lifelong.core.config import WakeConfig, SleepConfig
from lifelong.plugins.base import LifelongLearnerPlugin
from lifelong.util import shuffle_tensor, shuffle_tensors

class LifelongLearner:

    def __init__(
        self, env_names: List[str], 
        wake_learner_algo_instantiator: Callable[[str], ApexDQN], 
        sleep_algo_instantiator: Callable[[str, dict], DQN],
        sleep_buffer: ObservationBuffer,
        wake_buffer_collector_fn: Callable[[Algorithm, int], List[SampleBatch]],
        wake_config: WakeConfig = WakeConfig(),
        sleep_config: SleepConfig = SleepConfig(),
        sleep_logdir: str = os.path.join("logs", "sleep"),
        device: str = "cpu"
    ):
        self.env_names = env_names
        self.wake_baseline_mean_rewards = [0 for _ in env_names]

        self.wake_learner_algo_instantiator = wake_learner_algo_instantiator
        self.sleep_algo_instantiator = sleep_algo_instantiator
        self.sleep_policy_model: torch.nn.Module = None

        self.sleep_buffer = sleep_buffer

        self.wake_buffer_collector_fn = wake_buffer_collector_fn
        self.wake_buffer_obs_tensor: Tensor = Tensor()

        self.wake_config = wake_config
        self.sleep_config = sleep_config

        self.eval_tb_writer = SummaryWriter(os.path.join(sleep_logdir, "eval"))
        self.global_sleep_epoch = 0
        
        self.plugins: List[LifelongLearnerPlugin] = []
        self.device = device

    def learn(self):

        for plugin in self.plugins:
            plugin.before_learn(self)
        
        self.sleep_policy_model: torch.nn.Module = self.sleep_algo_instantiator(self.env_names[0], self.sleep_config.model_config).get_policy().model
        self.sleep_policy_model = self.sleep_policy_model.to(self.device)

        for env_i, env_name in enumerate(self.env_names):

            print(f" --- TASK {env_i+1}: {env_name} --- ")
            
            wake_learner = self.wake_learner_algo_instantiator(env_name)
            loaded_chkpt = False
            if os.path.exists(os.path.join("test_chkpts", env_name)):
                print("Found checkpoint for current environment, loading...")
                wake_learner.load_checkpoint(os.path.join("test_chkpts", env_name))
                loaded_chkpt = True

            best_chkpt_weights, best_mean_reward = self._learn_env(env_i, env_name, wake_learner, loaded_chkpt=loaded_chkpt)

            wake_policy = wake_learner.get_policy("default_policy")
            wake_policy.set_weights(best_chkpt_weights)
            wake_policy_model: torch.nn.Module = wake_policy.model.to(self.device)

            path = os.path.join("chkpts", env_name)
            os.makedirs(path, exist_ok=True)
            wake_learner.save_checkpoint(path)

            # collect observations from wake buffer
            start_time = time.time()
            print(f"Collecting task {env_i+1} buffer...", end="", flush=True)

            sample_batches = self.wake_buffer_collector_fn(wake_learner, self.sleep_config.buffer_capacity)

            # IDEA: have additional phase where sleep policy adjusts to new distribution of examples from prev. tasks generated
            # by the generative replay buffer, to help mitigate effects of including new task in the set of tasks that the gen. model has to accomodate
            # (generated examples from prev. tasks may look different after training gen. model to also generate examples from last task, which will
            # impact the self-supervised knowledge distillation phase). also maybe necessary after every new task to account for difference between ground truth
            # observations and generated ones.
            obs_batches = []
            for sample_batch in sample_batches:
                obs = torch.from_numpy(sample_batch[sample_batch.OBS])
                obs_batches.append(obs)

            del sample_batches
            # reorganize dimensions so the channel dimension is second ([NxCxHxW]) as opposed to last ([NxHxWxC])
            self.wake_buffer_obs_tensor = torch.unsqueeze(torch.cat(obs_batches, dim=0), 1)
            self.wake_buffer_obs_tensor = torch.squeeze(torch.swapdims(self.wake_buffer_obs_tensor, 1, 4)).to("cpu")
            wake_learner.stop()
            print(f'finished in {round(time.time()-start_time, 2)}s')

            # get eval score for best wake model to serve as baseline for sleep model
            start_time = time.time()
            print(f"Evaluating best wake model checkpoint for baseline mean reward...", end="", flush=True)
            eval_results = self._eval_model(wake_policy_model, self.wake_config.model_config, env_name)
            self.wake_baseline_mean_rewards[env_i] = eval_results["evaluation"]["episode_reward_mean"]
            print(f"finished in {round(time.time()-start_time, 2)}s. Mean reward was {self.wake_baseline_mean_rewards[env_i]}.\n")

            should_sleep = not self.sleep_config.copy_first_task_weights or env_i != 0
            for plugin in self.plugins:
                plugin.before_sleep(env_i, self, should_sleep)

            if should_sleep:
                self._sleep(env_i, wake_policy_model)
            else:
                # if this is first task / env, just set sleep policy to be current wake policy's best chkpt
                self.sleep_policy_model.load_state_dict(wake_policy_model.state_dict())

            for plugin in self.plugins:
                plugin.after_sleep(env_i, self, should_sleep)

            print("")

    def _learn_env(self, env_i, env_name, wake_learner: ApexDQN, loaded_chkpt: bool = False):

        best_chkpt_weights = None
        best_mean_reward = -math.inf

        start_time = time.time()
        print(f"Training wake model on task {env_i+1} ({env_name})...", end="", flush=True)

        timestep_threshold = None if loaded_chkpt else self.wake_config.timesteps_per_env
        while True:
            train_results = wake_learner.train()

            mean_reward = train_results["episode_reward_mean"]
            n_ts = train_results["num_agent_steps_sampled"]
            if mean_reward > best_mean_reward or best_chkpt_weights is None:
                # chkpt best model weights
                best_chkpt_weights = ray.get(wake_learner.curr_learner_weights)["default_policy"]  # curr_learner_weights is objectref to remote obj
                best_mean_reward = mean_reward

            if timestep_threshold is None:
                timestep_threshold = n_ts + self.sleep_config.buffer_capacity

            elif train_results["num_agent_steps_sampled"] >= timestep_threshold:
                break

        print(f"finished in {'{}'.format(str(timedelta(seconds=round(time.time()-start_time))))}", flush=True)

        print(f"Wake learner's highest mean reward on {env_name} was {round(best_mean_reward, 6)}.")

        return best_chkpt_weights, mean_reward


    def _sleep(self, wake_env_i: int, wake_policy_model: torch.nn.Module) -> (List[List[float]], List[float]):

        start_time = time.time()
        print("Executing sleep phase...", end="", flush=True)

        n_wake_exs = len(self.wake_buffer_obs_tensor)
        obs_shape = self.wake_buffer_obs_tensor.shape[1:]

        # knowledge distillation w/ self-supervised learning
        self.sleep_policy_model.train()
        wake_policy_model.eval()

        env_eval_hists = [[] for _ in range(len(self.env_names))]
        loss_hist = []

        def desc_str() -> str:
            hist_strs = [f'{name}: {env_eval_hists[i]}' for i, name in enumerate(self.env_names)]
            desc = "Reward hists:\n" + \
                '\n'.join(hist_strs)
            return desc
        
        eval_interval = self.sleep_config.sleep_eval_interval
        reinit_sleep_model = self.sleep_config.reinit_sleep_model
        pbar = qqdm(range(self.sleep_config.sleep_epochs), desc=desc_str())

        # preserve copy of sleep policy model prior to knowledge distillation for self-supervised learning targets
        prev_sleep_policy_model: torch.nn.Module = self.sleep_algo_instantiator(self.env_names[wake_env_i], self.sleep_config.model_config).get_policy().model.to(self.device)
        prev_sleep_policy_model.load_state_dict(self.sleep_policy_model.state_dict())

        if reinit_sleep_model:
            self.sleep_policy_model: torch.nn.Module = self.sleep_algo_instantiator(self.env_names[wake_env_i], self.sleep_config.model_config).get_policy().model.to(self.device)
            self.sleep_policy_model.train()

        # TODO: try persisting the optimiser
        loss_fn = torch.nn.functional.mse_loss
        optim = torch.optim.Adam(self.sleep_policy_model.parameters(), 0.0001)

        amp_enabled = True
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        
        # rllib policy models expect [N x H x W x C] as opposed to [N x C x H x W]
        pr_obs_tensor = None
        if wake_env_i > 0:
            pr_obs_tensor = self.sleep_buffer.sample(n_wake_exs, storage_device="cpu")
            pr_obs_tensor = torch.squeeze(torch.swapdims(torch.unsqueeze(pr_obs_tensor, 4), 1, 4), 1).to(self.device)
        wake_buffer_obs_tensor = torch.squeeze(torch.swapdims(torch.unsqueeze(self.wake_buffer_obs_tensor, 4), 1, 4), 1).to(self.device)

        # precalculate targets
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
            dummy_embeddings, _ = wake_policy_model.forward({"obs": wake_buffer_obs_tensor[:2]}, None, None)
            adv_out_shape = wake_policy_model.advantage_module(dummy_embeddings).shape[1:]  # ignore batch dimension

            # knowldege distillation targets (using wake buffer observations)
            kd_adv_targets = torch.zeros((n_wake_exs, *adv_out_shape))
            kd_value_targets = torch.zeros((n_wake_exs, 1))

            bs = 1_000
            i = 0
            while (i := i + bs) < n_wake_exs:
                kd_target_embeddings, _ = wake_policy_model.forward({"obs": wake_buffer_obs_tensor[i:i+bs]}, None, None)
                kd_adv_targets[i:i+bs] = wake_policy_model.advantage_module(kd_target_embeddings)
                kd_value_targets[i:i+bs] = wake_policy_model.value_module(kd_target_embeddings)

            # (pseudo-)rehearsal targets (using sleep replay buffer)
            n_pr = len(pr_obs_tensor)
            pr_adv_targets = torch.zeros((n_pr, *adv_out_shape))
            pr_value_targets = torch.zeros((n_pr, 1))

            if pr_obs_tensor is not None:
                i = 0
                while (i := i + bs) < n_pr:
                    pr_target_embeddings, _ = prev_sleep_policy_model.forward({"obs": pr_obs_tensor[i:i+bs]}, None, None)
                    pr_adv_targets[i:i+bs] = prev_sleep_policy_model.advantage_module(pr_target_embeddings)
                    pr_value_targets[i:i+bs] = prev_sleep_policy_model.value_module(pr_target_embeddings)


        # if self.sleep_config.reconstruct_wake_obs_before_kd:
        #     old_wake_buffer_obs_tensor = wake_buffer_obs_tensor
        #     wake_buffer_obs_tensor = torch.zeros((len(old_wake_buffer_obs_tensor, 4, 84, 84)))

        kd_batch_prop = 1/(wake_env_i+1)  # so that smapling is uniform across all tasks seen so far (assuming sleep replay buffer samples uniformly)
        alpha = self.sleep_config.kd_alpha
        pr_batch_size = math.floor(self.sleep_config.batch_size*(1-kd_batch_prop))
        kd_batch_size = math.ceil(self.sleep_config.batch_size*kd_batch_prop)
        for epoch in pbar:
            
            # periodically eval sleep model on all environments
            if epoch % eval_interval == 0:

                if epoch==0 and not self.sleep_config.eval_at_start_of_sleep:
                    continue

                scalar_dict = {}
                for eval_i in range(wake_env_i+1):
                    env_name = self.env_names[eval_i]
                    eval_results = self._eval_model(self.sleep_policy_model, self.sleep_config.model_config, env_name)
                    #TODO print evela results to se where the bottleneck is !!

                    mean_rew = eval_results["evaluation"]["episode_reward_mean"]
                    env_eval_hists[eval_i].append(mean_rew)
                    scalar_dict[env_name] = mean_rew / self.wake_baseline_mean_rewards[eval_i]

                if wake_env_i > 0:
                    self.eval_tb_writer.add_scalars("mean_reward", scalar_dict, global_step=self.global_sleep_epoch)
                else:
                    self.eval_tb_writer.add_scalar("mean_reward", scalar_dict[self.env_names[0]], global_step=self.global_sleep_epoch)

            total_loss = 0

            # shuffle observations from both wake buffer and sleep buffer
            wake_buffer_obs_tensor, kd_value_targets, kd_adv_targets = shuffle_tensors(
                self.wake_buffer_obs_tensor,
                kd_value_targets,
                kd_adv_targets
            )

            if pr_obs_tensor is not None:
                pr_obs_tensor, pr_value_targets, pr_adv_targets = shuffle_tensors(
                    pr_obs_tensor,
                    pr_value_targets,
                    pr_adv_targets
                )

            for batch_i in range(len(self.wake_buffer_obs_tensor) // kd_batch_size):
                
                pseudo_rehearsal_obs = pr_obs_tensor[ss_inds := slice(pr_batch_size*batch_i, pr_batch_size*(batch_i+1))] if pr_obs_tensor is not None else None
                knowledge_dist_obs = wake_buffer_obs_tensor[kd_inds := slice(kd_batch_size*batch_i, kd_batch_size*(batch_i+1))]

                # pseudo-rehearsal
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    pr_loss = 0
                    if pseudo_rehearsal_obs is not None:
                        pr_pred_embeddings, _ = self.sleep_policy_model.forward({"obs": pseudo_rehearsal_obs}, None, None)
                        pr_adv_preds = self.sleep_policy_model.advantage_module(pr_pred_embeddings)
                        pr_value_preds = self.sleep_policy_model.value_module(pr_pred_embeddings)

                        pr_loss = loss_fn(pr_adv_preds, pr_adv_targets[ss_inds]) + loss_fn(pr_value_preds, pr_value_targets[ss_inds])

                    # knowledge distillation
                    kd_pred_embeddings, _ = self.sleep_policy_model.forward({"obs": knowledge_dist_obs}, None, None)
                    kd_adv_preds = self.sleep_policy_model.advantage_module(kd_pred_embeddings)
                    kd_value_preds = self.sleep_policy_model.value_module(kd_pred_embeddings)

                    kd_loss = loss_fn(kd_adv_preds, kd_adv_targets[kd_inds]) + loss_fn(kd_value_preds, kd_value_targets[kd_inds])

                combined_loss = alpha*kd_loss + (1-alpha)*pr_loss
                total_loss += combined_loss.cpu().item()

                optim.zero_grad()

                scaler.scale(combined_loss).backward()
                scaler.step(optim)
                scaler.update()

            if (epoch+1) % eval_interval == 0:
                loss_hist.append(round(total_loss / (batch_i+1), 6))
            
            pbar.set_description(desc_str())
            self.global_sleep_epoch += 1

        wake_policy_model.train()

        print(f"finished in {'{}'.format(str(timedelta(seconds=round(time.time()-start_time))))}")

        print("Distillation loss over sleeping epochs:", loss_hist)
        print(f"Environment mean rewards over sleeping (interval of {eval_interval}):", flush=True)
        for env_name, eval_hist in zip(self.env_names, env_eval_hists):
            print(f" - {env_name}: {eval_hist}")

        return env_eval_hists, loss_hist
    
    def _eval_model(self, model: torch.nn.Module, model_config: dict, env_name: str):
        eval_algo = self.sleep_algo_instantiator(env_name, model_config)
        eval_algo.get_policy().set_weights(model.state_dict())
        eval_results = eval_algo.evaluate()
        eval_algo.stop()

        return eval_results
            