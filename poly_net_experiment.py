from src.real_cube_dataset import RealCubeDataset
from src.trainer import Trainer

import random
import numpy as np
import torch
import warp as wp
import os

def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

if __name__ == "__main__":
    set_seed(42)

    device = wp.get_preferred_device()
    print(f"Running on: {device}")

    data_path = os.path.expanduser('~/projects/contact-nets/data/tosses_processed/')
    dataset = RealCubeDataset(data_path, device.alias)
    # contact nets doubles the tosses which kinda makes sense since we use paired data
    # but I think it removes confusion if you keep it constant
    dataset.load(splits=[50, 30, 20], num_tosses=200, noise=0.4)


    trainer = Trainer(dataset)
    trainer.train()
