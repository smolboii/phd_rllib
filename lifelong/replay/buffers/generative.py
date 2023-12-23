from typing import Callable
import math
import os
from collections import defaultdict

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from qqdm import qqdm

from vae import VAE
from .base import ObservationBuffer
from .raw import RawObservationBuffer
from lifelong.util import shuffle_tensor


class GenerativeObservationBuffer(ObservationBuffer):

    def __init__(
            self, 
            gen_model_creator: Callable[[], VAE], 
            observation_shape: tuple, 
            n_epochs_per_train: int, 
            train_batch_size: int,
            kl_beta_annealing_fn: Callable[[int], float] = lambda curr_iters: 0.0001, 
            log_interval: int = 5,
            log_dir: str = "gen_buffer",
            reset_model_before_train: bool = True,
            last_task_loss_weighting: float = None,
            gen_obs_batch_prop: float = None,
            raw_buffer_capacity_per_task: int = 1_000,
            raw_buffer_obs_batch_prop: float = 0.1,
            device: str = "cpu"
        ):
        super().__init__(observation_shape, device=device)

        self.gen_model = gen_model_creator().to(device)
        self.gen_model_creator = gen_model_creator

        self.n_epochs_per_train = n_epochs_per_train
        self.train_batch_size = train_batch_size

        self.kl_beta_annealing_fn = kl_beta_annealing_fn
        self.reset_model_before_train = reset_model_before_train

        self.last_task_loss_weighting = last_task_loss_weighting

        self.gen_obs_batch_prop = gen_obs_batch_prop

        self.raw_buffer_capacity_per_task = raw_buffer_capacity_per_task
        self.raw_buffer_obs_batch_prop = raw_buffer_obs_batch_prop
        self.raw_buffer = None
        if raw_buffer_capacity_per_task > 0:
            self.raw_buffer = RawObservationBuffer(observation_shape, capacity=raw_buffer_capacity_per_task, share_between_tasks=False)

        os.mkdir(log_dir)
        self.log_dir = log_dir
        self.log_interval = log_interval
        self.tb_logger = SummaryWriter(log_dir)


    def train(self, env_i: int, wake_buffer_tensor: Tensor, logging: bool = True):

        wake_buffer_size = len(wake_buffer_tensor)
        wake_buffer_tensor = wake_buffer_tensor.to(self.device)

        prev_tasks_raw_obs = None
        if (self.raw_buffer is not None) and len(self.raw_buffer.obs_buffers) > 0:
            prev_tasks_raw_obs = self.raw_buffer.sample(min(len(self.raw_buffer.obs_buffers) * self.raw_buffer.capacity, wake_buffer_size), storage_device=self.device)

        # generate a dataset of samples representative of earlier tasks
        prev_tasks_gen_obs = self.sample(wake_buffer_size)
        gen_obs_batch_prop = self.gen_obs_batch_prop if self.gen_obs_batch_prop is not None else (1 - 1/(env_i+1))  # uniform spread if not specified
        if env_i == 0:
            gen_obs_batch_prop = 0  # no self-supervised exs for learning / sleeping first task
        
        if self.reset_model_before_train:
            self.gen_model = VAE(self.gen_model.in_channels, self.gen_model.latent_dim, desired_hw=self.gen_model.desired_hw)
            self.gen_model.to(self.device)

        # train generative replay model
        pbar = qqdm(range(self.n_epochs_per_train))
        optim = torch.optim.Adam(self.gen_model.parameters(), 0.001)

        amp_enabled = True
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

        self.gen_model.train()

        curr_train_iters = 0
        raw_obs_bs = math.ceil(self.raw_buffer_obs_batch_prop * self.train_batch_size)
        wake_obs_bs = math.ceil((1-gen_obs_batch_prop) * self.train_batch_size)
        gen_obs_bs = math.ceil(gen_obs_batch_prop * self.train_batch_size)
        ltl_weight = self.last_task_loss_weighting if self.last_task_loss_weighting is not None else 1/(env_i+1)
        for epoch in pbar:

            wake_buffer_tensor = shuffle_tensor(wake_buffer_tensor).to(self.device)
            if prev_tasks_raw_obs is not None:
                prev_tasks_raw_obs = shuffle_tensor(prev_tasks_raw_obs).to(self.device)
            if prev_tasks_gen_obs is not None:
                prev_tasks_gen_obs = shuffle_tensor(prev_tasks_gen_obs).to(self.device)

            if logging and (epoch+1) % self.log_interval == 0:
                plt.gray()
                self._save_samples(wake_buffer_tensor, 10, epoch, env_i)

            # NOTE: could KL annealing have an adverse affect on learning from the self-supervised examples somehow? idk.
            # why? i dont remember

            total_loss = 0
            total_recons_loss = 0
            total_kld_loss = 0
            for batch_i in range(wake_buffer_size // wake_obs_bs):
                # update kl weighting
                self.gen_model.kld_beta = self.kl_beta_annealing_fn(curr_train_iters)

                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    gt_obs = wake_buffer_tensor[batch_i*wake_obs_bs : (batch_i+1)*wake_obs_bs]
                    reconstructed_gt_obs_tuple = self.gen_model.forward(gt_obs)
                    loss_dict = self.gen_model.loss_function(*reconstructed_gt_obs_tuple)

                    # generate examples from prev. tasks using stored copy of gen replay model to preserve knowledge
                    ss_loss_dict = defaultdict(lambda: 0)
                    if env_i != 0 and ((prev_tasks_raw_obs is not None) or (prev_tasks_gen_obs is not None)):  # nothing to self-supervise for first task
                        
                        start_i = (batch_i * gen_obs_bs) % len(prev_tasks_gen_obs)
                        ss_obs = prev_tasks_gen_obs[start_i : start_i + gen_obs_bs] if gen_obs_bs > 0 else torch.zeros((0, *self.observation_shape), device=self.device)
                        
                        if (prev_tasks_raw_obs is not None) and raw_obs_bs > 0:
                            # concatenate some stored raw observations from prev. tasks
                            start_i = (batch_i*raw_obs_bs) % len(prev_tasks_raw_obs)
                            ss_obs = torch.cat((ss_obs, prev_tasks_raw_obs[start_i : start_i+raw_obs_bs]))

                        reconstructed_ss_obs_tuple = self.gen_model.forward(ss_obs)
                        ss_loss_dict = self.gen_model.loss_function(*reconstructed_ss_obs_tuple)

                loss = ltl_weight*loss_dict["loss"] + (1-ltl_weight)*ss_loss_dict["loss"]

                optim.zero_grad()

                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()

                total_loss += loss.cpu().item()
                total_recons_loss += (loss_dict["recons_loss"] + ss_loss_dict["recons_loss"]).cpu().item()
                total_kld_loss += (loss_dict["kld_loss"] + ss_loss_dict["kld_loss"]).cpu().item()

                curr_train_iters += 1

            if (epoch+1) % self.log_interval == 0:
                self.tb_logger.add_scalar(f"task_{env_i+1}/loss", round(total_loss / (batch_i+1), 8), epoch+1)
                self.tb_logger.add_scalar(f"task_{env_i+1}/reconstruction_loss", round(total_recons_loss / (batch_i+1), 8), epoch+1)
                self.tb_logger.add_scalar(f"task_{env_i+1}/KLD_loss", round(total_kld_loss / (batch_i+1), 8), epoch+1)
                
        if self.raw_buffer is not None:
            # add some observations from last task to the raw observation buffer for future use
            self.raw_buffer.add_observations(wake_buffer_tensor[:self.raw_buffer_capacity_per_task])

    def sample(self, count: int, batch_size: int = 2_500, storage_device: str = None) -> Tensor:

        # samples the required count of observations from the generative buffer, splitting into batches of 'batch_size' to help prevent "out of memory" errors
        self.gen_model.eval()

        storage_device = storage_device if storage_device is not None else self.device
        with torch.no_grad():

            prev_tasks_gen_obs = torch.zeros((count, *self.observation_shape), device=self.device)

            i = 0
            while i < count:
                adj_bs = min(batch_size, count-i)  # 'batch_size' might not evenly divide 'count'
                prev_tasks_gen_obs[i:i+adj_bs] = self.gen_model.sample(adj_bs, self.device).to(storage_device)
                i += batch_size

        return prev_tasks_gen_obs
    
    def reset(self) -> None:
        self.gen_model = self.gen_model_creator().to(self.device)
        if self.raw_buffer is not None:  # if it's None there is nothing to reset
            self.raw_buffer = RawObservationBuffer(self.observation_shape, capacity=self.raw_buffer.capacity, share_between_tasks=False)

    def _save_samples(self, observations: Tensor, n: int, epoch: int, env_i: int):

        task_dir_name = f"task_{env_i+1}"
        full_dir = os.path.join(self.log_dir, "samples", task_dir_name)
        os.makedirs(full_dir, exist_ok=True)
        with torch.no_grad():
            for i_ in range(n):
                gt_obs = torch.unsqueeze(observations[i_], 0)  # add back batch dim with unsqueeze
                recons_obs = self.gen_model.forward(gt_obs)[0]
                gt_obs = gt_obs.squeeze().cpu().detach().numpy()
                recons_obs = recons_obs.squeeze().cpu().detach().numpy()
                sampled_obs = self.gen_model.sample(1, current_device=self.device)[0].cpu().detach().numpy()
                flattened_frames = np.zeros((84*3, 84*4))
                for j_ in range(4):
                    flattened_frames[:84,j_*84:(j_+1)*84] = gt_obs[j_]
                    flattened_frames[84:84*2,j_*84:(j_+1)*84] = recons_obs[j_]
                    flattened_frames[84*2:, j_*84:(j_+1)*84] = sampled_obs[j_]
                plt.imsave(os.path.join(full_dir, f"epoch_{epoch}-{i_+1}.png"), flattened_frames)

def cyclic_anneal_creator(total_train_iters: int, n_cycles: int, R: float, start_beta: float = 0, end_beta: float = 1):
    """ Creates a cyclic annealing function for the KL weighting term (beta) in the VAE loss function

    Args:
        total_train_iters (int): Number of training iterations (minibatches) the VAE will be trained for
        n_cycles (int): Number of cycles to have over the course of training
        R (float): Proportion of each cycle to anneal over (e.g., R=0.5 means beta will reach 'end_beta' at the half-way point of each cycle)
        start_beta (float): Value of beta that each cycle starts at
        end_beta (float): Value of beta that is reached each cycle after fully annealed (prop. of cycle dictated by R)
    """

    assert 0 < R < 1, "R must be in (0, 1)"

    cycle_len = total_train_iters // n_cycles
    def cyclic_anneal_fn(curr_train_iters: int):
        cycle_train_iters = curr_train_iters % cycle_len
        kl_beta = start_beta + (end_beta - start_beta) * min(1, cycle_train_iters / (cycle_len*R))
        return kl_beta
    
    return cyclic_anneal_fn

