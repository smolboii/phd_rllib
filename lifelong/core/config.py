
class SleepConfig:

    def __init__(
        self,
        sleep_epochs: int = 100,
        reinit_sleep_model: bool = False,
        sleep_eval_interval: int = 20,
        eval_at_start_of_sleep: bool = False,
        copy_first_task_weights: bool = True,
        batch_size: int = 256,
        buffer_capacity: int = 100_000,
        reconstruct_wake_obs_before_kd: bool = False,
        kd_alpha: float = 0.55,
        model_config: dict = dict()
    ):
        self.sleep_epochs = sleep_epochs
        self.reinit_sleep_model = reinit_sleep_model
        self.sleep_eval_interval = sleep_eval_interval
        self.eval_at_start_of_sleep = eval_at_start_of_sleep
        self.copy_first_task_weights = copy_first_task_weights
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.reconstruct_wake_obs_before_kd = reconstruct_wake_obs_before_kd
        self.kd_alpha = kd_alpha
        self.model_config = model_config

class WakeConfig:

    def __init__(
        self,
        timesteps_per_env: int = 20_000_000,
        model_config: dict = dict()
    ):
        self.timesteps_per_env = timesteps_per_env
        self.model_config = model_config