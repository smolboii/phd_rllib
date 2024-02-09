from logging import Logger, getLogger
from operator import attrgetter
from typing import List, Union, Callable
from copy import deepcopy, copy
import os
import logging
import math

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from ray.rllib.algorithms import Algorithm
from matplotlib import pyplot
import mnist
import cv2

from lifelong.core.lifelong_learner import LifelongLearner
from lifelong.models.wrappers.base import ModelWrapper
from .base import LifelongLearnerPlugin
from lifelong.util import shuffle_tensors

class WarmStarterPlugin(LifelongLearnerPlugin):

    def __init__(self, should_warmstart_callback: Callable[[LifelongLearner], bool], logger: Logger = logging.getLogger("")):
        super().__init__(logger)

        self.should_warmstart_callback = should_warmstart_callback

    def before_learn_env(self, env_name: str, wake_learner: Algorithm, lifelong_learner: LifelongLearner):
        super().before_learn_env(env_name, wake_learner, lifelong_learner)

        if self.should_warmstart_callback(lifelong_learner):
            self.logger.log(logging.INFO, f"Using current sleep model as starting point for wake model (warm starting) on {env_name}")
            wake_learner.get_policy().model.load_state_dict(lifelong_learner.sleep_model_wrapper.model.state_dict())
