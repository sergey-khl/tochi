from torch.utils.data import DataLoader
import warp as wp
import numpy as np
import math
import torch
from dataclasses import dataclass

from src.loss import SurrogateConfig3D

@dataclass
class Block3DTraining:
    learn_normal: bool = True
    learn_tangent: bool = True

    # 'poly' or 'deepvertex' or 'deep'
    net_type: str = 'poly'

    surrogate_config: SurrogateConfig3D = SurrogateConfig3D(
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

    elastic: float = False
    device: str = 'cuda'
    lr: float = 1e-4
    scheduler_step_size: int = 30
    scheduler_gamma: float = 1.0
    wd: float = 1e-2
    noise: float = 0.0
    H: int = 256
    k: int = 8

    epochs: int = 100
    batch: int = 1

# TODO: cite contact
class Trainer:
    def __init__(self, dataset):
        wp.init()

        self.params = Block3DTraining

        self.dataset = dataset

    def train(self):
        dataloader = DataLoader(self.dataset.train, batch_size=self.params.batch, shuffle=True)

        for batch in dataloader:
            print(batch)
            break
            # bad_loss = self.training_step(optimizer, batch[0], surpress_printing)
            # if bad_loss: bad_losses += 1

