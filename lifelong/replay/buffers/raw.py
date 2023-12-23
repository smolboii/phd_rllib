from typing import Callable
import math
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor
from qqdm import qqdm

from .base import ObservationBuffer

class RawObservationBuffer(ObservationBuffer):

    def __init__(self, observation_shape: tuple, capacity: int = 100_000, share_between_tasks: bool = False, device: str = "cpu"):
        super().__init__(observation_shape, device)

        assert share_between_tasks is False, "sharing buffer between tasks not yet supported"

        self.capacity = capacity
        self.share_between_tasks = share_between_tasks
        self.obs_buffers = []

    def add_observations(self, observations: Tensor) -> int:
        # adds observations to buffer and returns how many were preserved
        #NOTE: should we copy the tensor before adding? idk
        self.obs_buffers.append(observations[:self.capacity].clone().to(self.device))

        if len(self.obs_buffers) > 1 and self.share_between_tasks:
            # prune buffers to keep within capacity
            largest_size = max(self.obs_buffers, key=lambda buffer: len(buffer))
            self.obs_buffers[-1] = self.obs_buffers[-1][:largest_size]

            total = sum([len(buffer) for buffer in self.obs_buffers])
            reduction_factor = self.capacity / total

            for i, buffer in enumerate(self.obs_buffers):
                new_len = math.floor(len(buffer) * reduction_factor)
                self.obs_buffers[i] = buffer[:new_len]

            print(f"NEW TOTAL SIZE: {sum([len(buffer) for buffer in self.obs_buffers])}")

    def sample(self, count: int, storage_device: str = None) -> Tensor:
        storage_device = storage_device if storage_device is not None else self.device

        n_buffers = len(self.obs_buffers)
        count_per_buffer = count // n_buffers
        sampled_obs_list = []

        for i, buffer in enumerate(self.obs_buffers):
            inds = torch.randint(0, len(buffer), (count_per_buffer,))
            sampled_obs_list.append(buffer[inds].to(storage_device))

        return torch.cat(sampled_obs_list, dim=0).to(storage_device)  # concatenate samples together and return