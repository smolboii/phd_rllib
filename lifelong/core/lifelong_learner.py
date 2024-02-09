import os
import math
import numpy as np
import logging
from copy import deepcopy

from datetime import timedelta
from typing import Dict, Any, List, Callable

import filtros
filtros.activate()

from qqdm import qqdm
from matplotlib import pyplot as plt
import torch
from torch import Tensor
from torch.nn import MSELoss
import ray
from torch.utils.tensorboard import SummaryWriter
from ray.train.torch import get_device
from ray.rllib.algorithms.dqn.dqn_torch_policy import build_q_model_and_distribution
from ray.rllib.algorithms.apex_dqn import ApexDQNConfig, ApexDQN

from lifelong.replay.buffers.base import ObservationBuffer
from lifelong.plugins.base import LifelongLearnerPlugin
from lifelong.util import shuffle_tensor, shuffle_tensors
from lifelong.core.loss import kld_distillation_loss_creator, softmax_mse_loss
from lifelong.config import LifelongLearnerConfig
from lifelong.callbacks.base import AlgorithmCallbackWrapper
from lifelong.models.wrappers.base import ModelWrapper

class LifelongLearner:

    def __init__(
        self,
        config: LifelongLearnerConfig,
        wake_algo_callback_wrapper: AlgorithmCallbackWrapper,
        sleep_algo_callback_wrapper: AlgorithmCallbackWrapper,
        sleep_buffer: ObservationBuffer,
        logger: logging.Logger = logging.getLogger(""),
        log_dir: str = "logs",
        device: str = "cpu"
    ):
        
        self.env_names = config.env_names
        self.wake_baseline_mean_rewards = [0 for _ in config.env_names]

        self.wake_algo_callback_wrapper = wake_algo_callback_wrapper
        self.sleep_algo_callback_wrapper = sleep_algo_callback_wrapper
        self.sleep_model_wrapper: ModelWrapper = sleep_algo_callback_wrapper.instantiate_model(self.env_names[0], device)

        self.sleep_buffer = sleep_buffer
        self.wake_buffer_obs_tensor: Tensor = Tensor()

        self.optim = torch.optim.Adam(self.sleep_model_wrapper.model.parameters(), lr=config.sleep.lr)
        self.config = config

        self.distillation_loss = None
        if self.config.sleep.distillation_type == "mse":
            self.distillation_loss = torch.nn.MSELoss(reduction='mean')
        elif self.config.sleep.distillation_type == "softmax_mse":
            self.distillation_loss = softmax_mse_loss
        elif self.config.sleep.distillation_type == "kld":
            self.distillation_loss = kld_distillation_loss_creator(temperature = self.config.sleep.softmax_temperature, reduction='mean')
        else:
            raise ValueError(f"invalid distillation loss type: {self.config.sleep.distillation_type}")

        self.logger = logger
        self.log_dir = log_dir
        self.eval_tb_writer = SummaryWriter(os.path.join(log_dir, "sleep", "eval"))
        self.global_sleep_epoch = 0

        self.loop_count = 0
        self.sleep_count = 0
        
        self.plugins: List[LifelongLearnerPlugin] = []
        self.device = device

    def learn(self):

        for plugin in self.plugins:
            plugin.before_learn(self)
            
        for loop_i in range(self.config.n_loops):
            for env_i, env_name in enumerate(self.env_names):
                
                self.logger.log(logging.INFO, f" --- TASK {env_i+1}: {env_name} --- ")

                for plugin in self.plugins:
                    plugin.before_instantiate_algorithm(env_name, self)
                wake_learner = self.wake_algo_callback_wrapper.instantiate_algorithm(env_name, os.path.join(self.log_dir, "wake", f"loop_{loop_i+1}", env_name))

                for plugin in self.plugins:
                    plugin.before_learn_env(env_name, wake_learner, self)
                best_chkpt_weights, best_mean_reward = self._learn_env(env_i, env_name, wake_learner)

                wake_policy = wake_learner.get_policy("default_policy")
                wake_policy.set_weights(best_chkpt_weights)
                wake_policy_model: ModelWrapper = self.wake_algo_callback_wrapper.wrap_model(wake_policy.model, self.device)

                path = os.path.join(self.log_dir, "chkpts", env_name)
                os.makedirs(path, exist_ok=True)
                wake_learner.save_checkpoint(path)

                # collect observations from wake buffer
                self.logger.log(logging.INFO, f"Collecting task {env_i+1} buffer...")

                self.wake_buffer_obs_tensor = self.wake_algo_callback_wrapper.buffer_collector(wake_learner, self.config.sleep.buffer_capacity)

                # get eval score for best wake model to serve as baseline for sleep model
                self.logger.log(logging.INFO, f"Evaluating best wake model checkpoint for baseline mean reward...")
                eval_results = self._eval_model(wake_policy_model.model, env_name)
                self.wake_baseline_mean_rewards[env_i] = eval_results["evaluation"]["episode_reward_mean"]
                self.logger.log(logging.INFO, f"Baseline mean reward is {eval_results['evaluation']['episode_reward_mean']}")

                if self.config.sleep.copy_first_task_weights and self.sleep_count == 0:
                    self.sleep_model_wrapper.model.load_state_dict(wake_policy_model.model.state_dict())

                should_sleep = not self.config.sleep.copy_first_task_weights or self.sleep_count != 0

                # precalculate targets for knowledge distillation and pseudo-rehearsal
                reshaped_wake_obs_tensor, pr_obs_tensor = None, None
                kd_targets, pr_targets = None, None
                if should_sleep:
                    # rllib policy models expect [N x H x W x C] as opposed to [N x C x H x W]
                    if self.sleep_count > 0:
                        pr_obs_tensor = self.sleep_buffer.sample(len(self.wake_buffer_obs_tensor), storage_device="cpu")
                        pr_obs_tensor = torch.squeeze(torch.swapdims(torch.unsqueeze(pr_obs_tensor, 4), 1, 4), 1).to(self.device)
                    reshaped_wake_obs_tensor = torch.squeeze(torch.swapdims(torch.unsqueeze(self.wake_buffer_obs_tensor, 4), 1, 4), 1).to(self.device)
                    kd_targets, pr_targets = self._pre_calc_targets(reshaped_wake_obs_tensor, pr_obs_tensor, wake_policy_model)

                for plugin in self.plugins:
                    plugin.before_sleep(env_name, self, should_sleep)

                if should_sleep:
                    self._sleep(env_i, wake_policy_model, reshaped_wake_obs_tensor, pr_obs_tensor, kd_targets, pr_targets)

                self.sleep_count += 1

                for plugin in self.plugins:
                    plugin.after_sleep(env_name, self, should_sleep)

            self.loop_count += 1

    def _learn_env(self, env_i: int, env_name: str, wake_learner: ApexDQN):

        best_chkpt_weights = None
        best_mean_reward = -math.inf

        self.logger.log(logging.INFO, f"Training wake model on task {env_i+1} ({env_name})...")

        timestep_threshold = None
        while True:
            train_results = wake_learner.train()

            mean_reward = train_results["episode_reward_mean"]
            n_ts = train_results["num_agent_steps_sampled"]
            if mean_reward > best_mean_reward or best_chkpt_weights is None:
                # chkpt best model weights
                best_chkpt_weights = ray.get(wake_learner.curr_learner_weights)["default_policy"]  # curr_learner_weights is objectref to remote obj
                best_mean_reward = mean_reward

            if timestep_threshold is None:
                # ensure we at least get enough experiences to fill the wake buffer
                timestep_threshold = max(n_ts + self.config.sleep.buffer_capacity, self.config.wake.timesteps_per_env)

            elif train_results["num_agent_steps_sampled"] >= timestep_threshold:
                break
        self.logger.log(logging.INFO, f"Wake learner's highest mean reward on {env_name} was {round(best_mean_reward, 6)}.")

        return best_chkpt_weights, mean_reward


    def _sleep(self, wake_env_i: int, wake_policy_model: ModelWrapper, wake_obs_tensor: Tensor, 
               pr_obs_tensor: Tensor, kd_targets: Tensor, pr_targets: Tensor) -> (List[List[float]], List[float]):

        self.logger.log(logging.INFO, "Executing sleep phase...")

        n_envs_seen = min(self.sleep_count+1, len(self.env_names))

        self.sleep_model_wrapper.model.train()
        wake_policy_model.model.to(self.device)
        wake_policy_model.model.eval()

        env_eval_hists = [[] for _ in range(len(self.env_names))]
        loss_hist = []

        scaler = torch.cuda.amp.GradScaler(enabled=self.config.sleep.amp_enabled)

        wake_env_name = self.env_names[wake_env_i]
        for plugin in self.plugins:
            plugin.before_knowledge_distillation(wake_env_name, self)

        def desc_str() -> str:
            hist_strs = [f'{name}: {env_eval_hists[i]}' for i, name in enumerate(self.env_names)]
            desc = "Reward hists:\n" + \
                '\n'.join(hist_strs)
            return desc
        
        pbar = qqdm(range(self.config.sleep.sleep_epochs), desc=desc_str())

        kd_batch_prop = 1 / n_envs_seen  # so that smapling is uniform across all tasks seen so far (assuming sleep replay buffer samples uniformly)
        alpha = self.config.sleep.kd_alpha if pr_obs_tensor is not None else 1
        pr_batch_size = math.floor(self.config.sleep.batch_size*(1-kd_batch_prop))
        kd_batch_size = math.ceil(self.config.sleep.batch_size*kd_batch_prop)
        eval_interval = self.config.sleep.eval_interval
        for epoch in pbar:

            for plugin in self.plugins:
                plugin.during_knowledge_distillation(epoch, wake_env_name, self)
            
            # periodically eval sleep model on all environments
            if epoch % self.config.sleep.eval_interval == 0 and (epoch > 0 or self.config.sleep.eval_at_start_of_sleep):

                scalar_dict = {}
                for eval_i in range(n_envs_seen):
                    env_name = self.env_names[eval_i]
                    eval_results = self._eval_model(self.sleep_model_wrapper.model, env_name)
                    #TODO print evela results to se where the bottleneck is !!

                    mean_rew = eval_results["evaluation"]["episode_reward_mean"]
                    env_eval_hists[eval_i].append(mean_rew)
                    scalar_dict[env_name] = mean_rew / self.wake_baseline_mean_rewards[eval_i]

                if n_envs_seen > 1:
                    self.eval_tb_writer.add_scalars("mean_reward", scalar_dict, global_step=self.global_sleep_epoch)
                else:
                    self.eval_tb_writer.add_scalar("mean_reward", scalar_dict[self.env_names[0]], global_step=self.global_sleep_epoch)

            total_loss = 0

            # shuffle observations from both wake buffer and pseudo-rehearsal buffer
            wake_obs_tensor, kd_targets = shuffle_tensors(
                wake_obs_tensor,
                kd_targets
            )

            if pr_obs_tensor is not None:
                pr_obs_tensor, pr_targets = shuffle_tensors(
                    pr_obs_tensor,
                    pr_targets
                )

            for batch_i in range(len(self.wake_buffer_obs_tensor) // kd_batch_size):
                
                pseudo_rehearsal_obs = pr_obs_tensor[pr_inds := slice(pr_batch_size*batch_i, pr_batch_size*(batch_i+1))] if pr_obs_tensor is not None else None
                knowledge_dist_obs = wake_obs_tensor[kd_inds := slice(kd_batch_size*batch_i, kd_batch_size*(batch_i+1))]

                with torch.cuda.amp.autocast(enabled=self.config.sleep.amp_enabled):
                    pr_loss = torch.zeros(1, device=self.device)
                    if pseudo_rehearsal_obs is not None:
                        pr_preds = self.sleep_model_wrapper.forward(pseudo_rehearsal_obs)
                        pr_loss = self.distillation_loss(pr_preds, pr_targets[pr_inds])

                    kd_preds = self.sleep_model_wrapper.forward(knowledge_dist_obs)
                    kd_loss = self.distillation_loss(kd_preds, kd_targets[kd_inds])

                combined_loss = alpha*kd_loss + (1-alpha)*pr_loss
                total_loss += combined_loss.detach().cpu().item()

                self.optim.zero_grad()

                scaler.scale(combined_loss).backward()
                scaler.step(self.optim)
                scaler.update()

            if (epoch+1) % eval_interval == 0:
                loss_hist.append(round(total_loss / (batch_i+1), 6))
            
            pbar.set_description(desc_str())
            self.global_sleep_epoch += 1

        wake_policy_model.model.train()

        self.logger.log(logging.INFO, "Distillation loss over sleeping epochs:", loss_hist)
        self.logger.log(logging.INFO, f"Environment mean rewards over sleeping (interval of {eval_interval}):")
        for env_name, eval_hist in zip(self.env_names, env_eval_hists):
            self.logger.log(logging.INFO, f" - {env_name}: {eval_hist}")
            
        return env_eval_hists, loss_hist
    
    def _pre_calc_targets(self, wake_obs_tensor: Tensor, pr_obs_tensor: Tensor, wake_policy_model: ModelWrapper):

        n_wake_exs = len(wake_obs_tensor)

        # precalculate targets
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.config.sleep.amp_enabled):
            out_shape = wake_policy_model.forward(wake_obs_tensor[:2]).shape[1:]  # ignore batch dimension

            # knowldege distillation targets (using wake buffer observations)
            kd_targets = torch.zeros((n_wake_exs, *out_shape), device=self.device)

            bs = 1_000
            i = 0
            while i < n_wake_exs:
                kd_targets[i:i+bs] = wake_policy_model.forward(wake_obs_tensor[i:i+bs])
                i += bs

            # (pseudo-)rehearsal targets (using sleep replay buffer)
            n_pr = len(pr_obs_tensor) if pr_obs_tensor is not None else 0
            pr_targets = torch.zeros((n_pr, *out_shape), device=self.device)

            if pr_obs_tensor is not None:
                i = 0
                while i < n_pr:
                    pr_targets[i:i+bs] = self.sleep_model_wrapper.forward(pr_obs_tensor[i:i+bs])
                    i += bs

        return (kd_targets, pr_targets)
    
    def _eval_model(self, model: torch.nn.Module, env_name: str):
        eval_algo = self.sleep_algo_callback_wrapper.instantiate_algorithm(env_name, "dummy_logs")
        eval_algo.get_policy().set_weights(model.state_dict())
        eval_results = eval_algo.evaluate()
        eval_algo.stop()

        return eval_results