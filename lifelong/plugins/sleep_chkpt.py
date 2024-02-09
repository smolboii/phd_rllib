
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

class SleepModelCheckpointerPlugin(LifelongLearnerPlugin):

    def __init__(self, chkpt_dir: str, logger: Logger = ...):
        super().__init__(logger)

        self.chkpt_dir = chkpt_dir

    def after_sleep(self, env_name: str, lifelong_learner: LifelongLearner, did_sleep: bool):
        super().after_sleep(env_name, lifelong_learner, did_sleep)

        sleep_model_weights = lifelong_learner.sleep_model_wrapper.model.state_dict()
        os.makedirs(self.chkpt_dir, exist_ok=True)
        torch.save(sleep_model_weights, os.path.join(self.chkpt_dir, f"L{lifelong_learner.loop_count}_S{lifelong_learner.sleep_count%len(lifelong_learner.env_names)}.pt"))

class SleepModelCheckpointLoaderPlugin(LifelongLearnerPlugin):

    def __init__(self, chkpt_path: str, logger: Logger = ...):
        super().__init__(logger)
        self.chkpt_path = chkpt_path
    
    def before_learn(self, lifelong_learner: LifelongLearner):
        super().before_learn(lifelong_learner)

        weights = torch.load(self.chkpt_path)
        lifelong_learner.sleep_model_wrapper.model.load_state_dict(weights)

