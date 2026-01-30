from typing import List
from dataclasses import dataclass, field
import torch
from torch import Tensor
import warp as wp

# TODO: maybe not the prettiest
device = wp.get_preferred_device()

@dataclass
class BlockEnvironmentConfig:
    # see https://github.com/DAIRLab/contact-nets/blob/main/README.md
    # METER_SCALE = 0.0524 # this is half box width
    meter_scale: float = 1.0 # this is half box width

    # see https://github.com/DAIRLab/contact-nets/blob/main/data/params_processed/experiment.json
    # To get the above ^^^ use these values
    ke: float = 1e4
    kd: float = 1e3
    kf: float = 50.0
    mu: float = 0.18
    restitution: float = 0.125 # TODO: in contact nets this is also hard coded to 0 in train.py. verify 
    sampling_dt: float = 0.006756756756756757
    sim_substeps: int = 10
    mass: float = 0.37

    # these are the ground truth that we should come to near the end of training
    # also make everything contgious for nice warp stuff
    vertices: Tensor = field(default_factory=lambda: torch.tensor([[-1, -1, -1, -1, 1, 1, 1, 1],
                                                   [-1, -1, 1, 1, -1, -1, 1, 1],
                                                   [1, -1, -1, 1, 1, -1, -1, 1]], device=device.alias).float().t().contiguous()
                             )
    ground_normal: Tensor = field(default_factory=lambda: torch.tensor([[0.0, 0, 1]], device=device.alias).contiguous())
    ground_tangent: Tensor = field(default_factory=lambda: torch.tensor([[1.0, 0, 0], [0, 1, 0]], device=device.alias).contiguous())

@dataclass
class BlockLossConfig:
    w_comp_n:               float = 0.0
    w_comp_t:               float = 0.0
    w_match:                float = 0.0
    w_cone:                 float = 0.0
    w_penetration_slack:    float = 0.0

    w_penetration:          float = 0.0
    w_config_grad_normal:   float = 0.0
    w_config_grad_tangent:  float = 0.0
    w_config_grad_perp:     float = 0.0
    w_st_estimate_pen:      float = 0.0
    w_st_estimate_normal:   float = 0.0
    w_st_estimate_tangent:  float = 0.0
    w_tangent_jac_d2:       float = 0.0

    w_contact_threshold:    float = -1.0

    robust_sqrt:            bool = False


@dataclass
class BlockConfig:
    device: str

    learn_normal: bool = True
    learn_tangent: bool = True

    # 'poly' or 'deepvertex' or 'deep'
    net_type: str = 'poly'
    elastic: float = False
    lr: float = 1e-4
    scheduler_step_size: int = 30
    scheduler_gamma: float = 1.0
    wd: float = 1e-2
    noise: float = 0.4
    H: int = 256
    k: int = 8

    seed: int = 42
    tosses: int = 200
    splits: List[int] = field(default_factory=lambda: [50, 30, 20])

    epochs: int = 100
    batch: int = 1

    # weights for loss
    loss_config: BlockLossConfig = BlockLossConfig(
            w_comp_n               = 1.0,
            w_comp_t               = 1.0,
            w_match                = 0.1,
            w_cone                 = 1.0,
            w_penetration_slack    = 1.0,

            w_penetration          = 0.0,
            w_config_grad_normal   = 0.3,
            w_config_grad_tangent  = 0.0,
            w_config_grad_perp     = 0.3,
            w_st_estimate_pen      = 0.0,
            w_st_estimate_normal   = 0.0,
            w_st_estimate_tangent  = 0.0,
            w_tangent_jac_d2       = 0.0,
            w_contact_threshold    = -0.0,

            robust_sqrt            = True
            )

    # descibes how env should react
    environment_config: BlockEnvironmentConfig = BlockEnvironmentConfig()

def load_config():
    # TODO: maybe do some overloading to make the config more interesting
    config = BlockConfig(device)

    return config
