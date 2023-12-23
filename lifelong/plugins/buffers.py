from lifelong.core.lifelong_learner import LifelongLearner
from .base import LifelongLearnerPlugin
from lifelong.replay.buffers.generative import GenerativeObservationBuffer
from lifelong.replay.buffers.raw import RawObservationBuffer

class RawObservationBufferPlugin(LifelongLearnerPlugin):
    def __init__(self, raw_obs_buffer: RawObservationBuffer):
        self.raw_obs_buffer = raw_obs_buffer

    def before_learn(self, lifelong_learner: LifelongLearner):
        super().before_learn(lifelong_learner)

        if not isinstance(lifelong_learner.sleep_buffer, RawObservationBuffer):
            raise Exception("RawObservationBufferPlugin must be used with a RawObservationBuffer sleep buffer.")

    def after_sleep(self, env_i: int, lifelong_learner: LifelongLearner, did_sleep: bool):
        super().after_sleep(env_i, lifelong_learner, did_sleep)
        self.raw_obs_buffer.add_observations(lifelong_learner.wake_buffer_obs_tensor)
        self.raw_obs_buffer.i = 0

class GenerativeObservationBufferPlugin(LifelongLearnerPlugin):

    def __init__(self, gen_obs_buffer: GenerativeObservationBuffer, logging: bool = True):
        self.gen_obs_buffer = gen_obs_buffer
        self.logging = logging

    def before_learn(self, lifelong_learner: LifelongLearner):
        super().before_learn(lifelong_learner)

        if not isinstance(lifelong_learner.sleep_buffer, GenerativeObservationBuffer):
            raise Exception("GenerativeObservationBufferPlugin must be used with a GenerativeObservationBuffer sleep buffer.")

    def after_sleep(self, env_i: int, lifelong_learner: LifelongLearner, did_sleep: bool):
        super().after_sleep(env_i, lifelong_learner, did_sleep)
        self.gen_obs_buffer.train(env_i, lifelong_learner.wake_buffer_obs_tensor, logging=self.logging)
