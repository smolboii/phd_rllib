import os
import logging
from typing import Union, Callable

from ray.rllib.algorithms import Algorithm

from lifelong.core.lifelong_learner import LifelongLearner
from .base import LifelongLearnerPlugin

class CheckpointLoaderPlugin(LifelongLearnerPlugin):

    def __init__(self, chkpt_dir: str, should_load_callback: Callable[[LifelongLearner], bool], logger: logging.Logger = logging.getLogger("")):
        super().__init__(logger)

        self.chkpt_dir = chkpt_dir
        self.should_load_callback = should_load_callback

    def before_learn_env(self, env_name: str, wake_learner: Algorithm, lifelong_learner: LifelongLearner):
        super().before_learn_env(env_name, wake_learner, lifelong_learner)

        if self.should_load_callback(lifelong_learner):
            if os.path.exists(os.path.join(self.chkpt_dir, env_name)):
                self.logger.log(logging.INFO, "Loading checkpoint for current environment...")
                wake_learner.load_checkpoint(os.path.join(self.chkpt_dir, env_name))
            else:
                self.logger.log(logging.ERROR, f"Could not find checkpoint for environment '{env_name}'.")
                raise Exception(f"Could not find checkpoint for current environment at path ''")
