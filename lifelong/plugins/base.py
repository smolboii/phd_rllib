from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lifelong.core.lifelong_learner import LifelongLearner


class LifelongLearnerPlugin:

    # base plugin class that can be extended and added to a lifelong learner to allow for customisable, additional behaviour
    # (has access to various hooks in the lifelong learner class)

    def __init__(self):
        pass

    def before_learn(self, lifelong_learner: 'LifelongLearner'):
        pass

    def before_sleep(self, env_i: int, lifelong_learner: 'LifelongLearner', will_sleep: bool):
        pass

    def after_sleep(self, env_i: int, lifelong_learner: 'LifelongLearner', did_sleep: bool):
        pass
