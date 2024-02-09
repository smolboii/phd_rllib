from typing import TYPE_CHECKING
import logging

from ray.rllib.algorithms import Algorithm

if TYPE_CHECKING:
    from lifelong.core.lifelong_learner import LifelongLearner


class LifelongLearnerPlugin:

    # base plugin class that can be extended and added to a lifelong learner to allow for customisable, additional behaviour
    # (has access to various hooks in the lifelong learner class)

    def __init__(self, logger: logging.Logger = logging.getLogger("")):
        self.logger = logger

    def before_learn(self, lifelong_learner: 'LifelongLearner'):
        pass

    def before_instantiate_algorithm(self, env_name: str, lifelong_learner: 'LifelongLearner'):
        pass

    def before_learn_env(self, env_name: str, wake_learner: Algorithm, lifelong_learner: 'LifelongLearner'):
        pass

    def during_learn_env(self, env_name: str, wake_learner: Algorithm, lifelong_learner: 'LifelongLearner'):
        pass

    def before_sleep(self, env_name: str, lifelong_learner: 'LifelongLearner', will_sleep: bool):
        pass

    def before_knowledge_distillation(self, env_name: str, lifelong_learner: 'LifelongLearner'):
        pass

    def during_knowledge_distillation(self, kd_epoch: int, env_name: str, lifelong_learner: 'LifelongLearner'):
        pass

    def after_sleep(self, env_name: str, lifelong_learner: 'LifelongLearner', did_sleep: bool):
        pass
