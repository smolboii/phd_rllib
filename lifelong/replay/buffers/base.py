from typing import Callable
import math

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor
from qqdm import qqdm

from vae import VAE

class ObservationBuffer:

    def __init__(self, observation_shape: tuple, device: str = "cpu"):
        self.observation_shape = observation_shape
        self.device = device

    def sample(self, count: int, storage_device: str = None) -> Tensor:
        raise NotImplementedError()
    
    def reset(self) -> None:
        raise NotImplementedError