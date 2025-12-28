import warp as wp
import warp.sim
import warp.sim.render
# import numpy as np
# import os
from environment import Environment
import torch

wp.init()

device = wp.get_preferred_device()
print(f"Running on: {device}")


env = Environment(device)

trajectory_length = 1000
wp_state = env.model.state()
wp_next_state = env.model.state()
wp_control = env.model.control()

for i in range(trajectory_length):
# while True:
    env.render()
    env.step(wp_state, wp_next_state, wp_control)

env.stop()
