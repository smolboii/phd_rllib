from dataclasses import dataclass
from typing import List


@dataclass
class WakeConfig:
    timesteps_per_env: int = 20_000_000,
    model_config: dict = dict()


@dataclass
class SleepConfig:
    sleep_epochs: int = 100,
    reinit_model_before_sleep: bool = False,
    sleep_eval_interval: int = 20,
    eval_at_start_of_sleep: bool = False,
    copy_first_task_weights: bool = True,
    batch_size: int = 256,
    buffer_capacity: int = 100_000,
    reconstruct_wake_obs_before_kd: bool = False,
    kd_alpha: float = 0.5,
    distlliaton_type: str = "mse",
    model_config: dict = dict()

@dataclass
class BufferConfig:

    type: str = "raw"

    ## raw buffer configs
    capacity: int = 100_000,
    share_between_tasks: bool = False

    ## generative buffer configs
    # general
    n_epochs_per_train: int = 100
    train_batch_size: int = 256
    log_interval: int = 5
    reset_model_before_train: bool = True
    pseudo_rehearsal_weighting: float = 0.5
    gen_obs_batch_prop: float = None
    raw_buffer_capacity_per_task: int = 1_000
    raw_buffer_obs_batch_prop: float = 0.1

    # vae
    latent_dims: int = 1024
    kld_beta = 0.0001
    log_var_clip_val = 5


@dataclass
class LifelongLearnerConfig:
    wake_config: WakeConfig = WakeConfig()
    sleep_config: SleepConfig = SleepConfig()
    buffer_config: BufferConfig = BufferConfig()

    exp_name: str = "default"
    env_names: List[str] = [
        "ALE/Asteroids-v5",
        "ALE/SpaceInvaders-v5",
        "ALE/Asterix-v5",
        "ALE/Alien-v5",
        "ALE/Breakout-v5"
    ]
    shuffle_envs: bool = False
    cuda_visible_devices: str = "0,1"