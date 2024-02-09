from logging import Logger, getLogger
from operator import attrgetter
from typing import List, Union, Callable
from copy import deepcopy, copy
import os
import logging
import math

import torch
from torch import nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from ray.rllib.algorithms import Algorithm
from ray.rllib.models.torch.visionnet import VisionNetwork
from gymnasium import Space
from ray.rllib.utils.typing import ModelConfigDict
from matplotlib import pyplot
import mnist
import cv2

from lifelong.core.lifelong_learner import LifelongLearner
from lifelong.models.wrappers.base import ModelWrapper
from lifelong.models.wrappers.dqn import DQNFeatureModelWrapper
from .base import LifelongLearnerPlugin
from lifelong.util import shuffle_tensors
from lifelong.models.dual_visionnet import DualVisionNetwork

class DualNetworkLoaderPlugin(LifelongLearnerPlugin):
    def __init__(self, should_load: Callable[[LifelongLearner], bool], logger: Logger = logging.getLogger("")):
        super().__init__(logger)

        self.should_load = should_load

    def before_instantiate_algorithm(self, env_name: str, lifelong_learner: LifelongLearner):
        super().before_instantiate_algorithm(env_name, lifelong_learner)

        if self.should_load(lifelong_learner):
            self.logger.log(logging.INFO, "Installing dual network...")

            # change model config so that the next wake algorithm has a dual network model
            lifelong_learner.config.wake.model_config["custom_model"] = DualVisionNetwork

            # save sleep model weights to disk to be loaded by wake learner
            # (tried passing it as a config dict entry, but this ran into issues with deserialisation for some reason)
            state_dict_dir = os.path.join("temp", "dual")
            os.makedirs(state_dict_dir, exist_ok=True)
            state_dict_path = os.path.join(state_dict_dir, "weights.pt")
            torch.save(lifelong_learner.sleep_model_wrapper.model.state_dict(), state_dict_path)

            lifelong_learner.config.wake.model_config["custom_model_config"] = {
                "state_dict_path": state_dict_path
            }

