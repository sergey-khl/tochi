import warp as wp
import warp.sim
import warp.sim.render
import os
from environment import Environment
from real_cube_dataset import RealTossesDataset, pad_collate
from torch.utils.data import DataLoader

from tqdm import trange

wp.init()

device = wp.get_preferred_device()
print(f"Running on: {device}")


env = Environment(device)

data_path = os.path.expanduser('~/projects/contact-nets/data/tosses_processed/')
dataset = RealTossesDataset(data_path)

# dataloader = DataLoader(
#     dataset, 
#     batch_size=16, 
#     shuffle=True, 
#     collate_fn=pad_collate
# )

for i in range(len(dataset)):
    trajectory = dataset[i]
    for sample in trajectory['full_state']:
        wp_state = env.state
        
        env.set_torch_state(sample)
        wp_next_state = env.state
        wp_control = env.control

        env.render()
        env.step(wp_state, wp_next_state, wp_control)

env.stop()
