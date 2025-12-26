import warp as wp
# import numpy as np
# import os
from environment import Environment

wp.init()

device = wp.get_preferred_device()
print(f"Running on: {device}")


model = Environment(device)
