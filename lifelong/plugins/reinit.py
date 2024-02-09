from operator import attrgetter
from typing import List, Union
from copy import deepcopy, copy
import logging

from lifelong.core.lifelong_learner import LifelongLearner

from .base import LifelongLearnerPlugin

class SleepModuleReinitialiserPlugin(LifelongLearnerPlugin):

    ALL_MODULES = 0

    def __init__(self, module_paths: Union[str, List[str], int], logger: logging.Logger = logging.getLogger("")):
        super().__init__(logger=logger)

        if type(module_paths) is str:
            module_paths = [module_paths]
        elif type(module_paths) is not list:
            if module_paths != self.ALL_MODULES:
                raise ValueError("Only non string / list of strings value accepted is the int literal ALL_MODULES to specify all modules.")

        self.module_paths = module_paths  # paths to the modules to reinitialise
        self.init_state_dicts: List[dict] = None

    def before_sleep(self, env_name: int, lifelong_learner: LifelongLearner, will_sleep: bool):
        # resets the specified modules back to their state before the first sleep period
        # TODO: add random reinitialisation also, as maybe resetting to same state everytime could be detrimental?

        super().before_sleep(env_name, lifelong_learner, will_sleep)
        
        model = lifelong_learner.sleep_model_wrapper.model

        if self.init_state_dicts is None:
            if self.module_paths == self.ALL_MODULES:
                self.init_state_dicts = deepcopy(model.state_dict())
            else:
                self.init_state_dicts = [deepcopy(attrgetter(path)(model).state_dict()) for path in self.module_paths]

        else:
            if self.module_paths == self.ALL_MODULES:
                model.load_state_dict(self.init_state_dicts)
            else:
                for path, state_dict in zip(self.module_paths, self.init_state_dicts):
                    attrgetter(path)(model).load_state_dict(state_dict)