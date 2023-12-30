from dataclasses import dataclass, field
from typing import List
import json

import dacite

## example for larger model by extending number of conv_filters

# larger_sleep_model_config = {
#     "conv_filters": [
#         [16, [8, 8], 4],
#         [32, [4, 4], 2],
#         [64, [3, 3], 2],
#         [128, [3, 3], 2],
#         [256, [3, 3], 2],
#         [256, [2, 2], 2],
#         [256, [1, 1], 1],
#     ]
# }

@dataclass
class WakeConfig:
    batch_size: int = 256
    lr: float = 0.00025 / 4
    timesteps_per_env: int = 20_000_000
    chkpt_dir: str = "pretrained_chkpts"
    model_config: dict = field(default_factory=dict)


@dataclass
class SleepConfig:
    batch_size: int = 256
    sleep_epochs: int = 100
    reinit_model_before_sleep: bool = False
    sleep_eval_interval: int = 20
    eval_at_start_of_sleep: bool = True
    copy_first_task_weights: bool = False
    buffer_capacity: int = 100_000
    reconstruct_wake_obs_before_kd: bool = False
    
    kd_alpha: float = 0.5
    distillation_type: str = "mse"
    softmax_temperature: float = 0.01

    model_config: dict = field(default_factory=dict)


@dataclass
class BufferConfig:

    type: str = "raw"

    ## raw buffer configs
    capacity: int = 100_000
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
    wake: WakeConfig = WakeConfig()
    sleep: SleepConfig = SleepConfig()
    buffer: BufferConfig = BufferConfig()

    exp_name: str = "default"
    env_names: List[str] = field(default_factory=lambda: [
        "ALE/Asteroids-v5",
        "ALE/SpaceInvaders-v5",
        "ALE/Asterix-v5",
        "ALE/Alien-v5",
        "ALE/Breakout-v5"
    ])
    shuffle_envs: bool = False
    cuda_visible_devices: str = "0,1"

def parse_config_json(path: str) -> LifelongLearnerConfig:
    try:
        with open("config.json") as cf_fp:
            cf_dict = json.loads(cf_fp.read())
            cf = dacite.from_dict(LifelongLearnerConfig, cf_dict, config=dacite.Config(strict=True))
    except FileNotFoundError:
        # just incase the config file is incorrectly named, we prevent execution unless there is a config.json file present
        raise Exception(f"Could not find config file with path: {path}")
    
    return cf