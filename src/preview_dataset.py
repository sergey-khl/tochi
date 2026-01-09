import warp as wp
import warp.sim
import warp.sim.render
import os
from environment import Environment
from real_cube_dataset import RealTossesDataset, pad_collate
from torch.utils.data import DataLoader
import torch
import time

from tqdm import trange

wp.init()

device = wp.get_preferred_device()
print(f"Running on: {device}")

data_path = os.path.expanduser('~/projects/contact-nets/data/tosses_processed/')
dataset = RealTossesDataset(data_path)

env = Environment(device)


for i in range(len(dataset)):
    trajectory = dataset[i]['full_state']
    for curr_sample, next_sample in zip(trajectory[0:-2], trajectory[1:]):
        env.set_torch_state(curr_sample, next_sample) # tells the environment about current and next data

        env.render()
        # env.step() # only step to see what warp thinks the simulation should do

env.stop()
