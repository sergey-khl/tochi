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
first_traj = dataset[0]['full_state']

env = Environment(device, first_traj[0][:7], first_traj[0][7:13])


# dataloader = DataLoader(
#     dataset, 
#     batch_size=16, 
#     shuffle=True, 
#     collate_fn=pad_collate
# )

# for i in range(len(dataset)):
trajectory = dataset[5]
    # for curr_sample, next_sample in zip(trajectory['full_state'][0:-2], trajectory['full_state'][1:]):
curr_sample = trajectory['full_state'][-1]
next_sample = trajectory['full_state'][-1]
        # # pos = [-1.4197,  -0.8784,   0.9944]
# pos = [-1.4197,  0.9944,  1.08]
        # # pos = [0, 1, 0]
# quat = [-0.3716,   0.9284,   -0.00375,   -0.001268]
# quat = [-0.00375,   -0.3716,   -0.001268,   0.9284]
        #  9.2837268e-01 -3.7508027e-03 -1.2681728e-03 -3.7162948e-01
        # # quat = [0,   0,   0,   1]
# vel = [0, 0, 0]
# ang_vel = [0, 0, 0]
# control = [0, 0, 0, 0, 0, 0]
# curr_sample = torch.tensor([*pos, *quat, *vel, *ang_vel, *control])
print(curr_sample)
        # curr_sample[1], curr_sample[2] = curr_sample[2], curr_sample[1]
        # curr_sample[1], curr_sample[2] = curr_sample[2], curr_sample[1]
        # next_sample[1], next_sample[2] = next_sample[2], next_sample[1]
env.set_torch_state(curr_sample, next_sample) # tells the environment about current and next data

env.render()
env.render()
        # env.step()

while True:
    pass
env.stop()
