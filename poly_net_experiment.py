from src.block_config import load_config
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
    params = load_config()
    set_seed(params.seed)

    print(f"Running on: {params.device}")

    data_path = os.path.expanduser('~/projects/contact-nets/data/tosses_processed/')
    dataset = RealCubeDataset(data_path, params.device.alias)
    # contact nets doubles the tosses which kinda makes sense since we use paired data
    # but I think it removes confusion if you keep it constant
    # dataset.load(splits=params.splits, num_tosses=params.tosses, noise=params.noise)
    dataset.load(splits=params.splits, num_tosses=params.tosses)

    trainer = Trainer(dataset)
    trainer.train()
