from torch.serialization import load
from torch.utils.data import DataLoader
import warp as wp
import numpy as np
import math
import torch
from tqdm import tqdm

from src.block_config import load_config
from src.contact_net import ContactNet

# TODO: cite contact
class Trainer:
    def __init__(self, dataset):
        wp.init()

        self.params = load_config()

        self.dataset = dataset

        # initialize contact nets model
        self.contact_net = ContactNet()


        self.optimizer = torch.optim.AdamW(self.contact_net.parameters(), lr=self.params.lr, weight_decay=self.params.wd)

    def train(self):
        dataloader = DataLoader(self.dataset.train, batch_size=self.params.batch, shuffle=True)

        for epoch in tqdm(range(self.params.epochs)):
            # TODO: convert to logging
            print(f"starting epoch {epoch}")
            for batch in dataloader:
                prev_config = batch[0][0, 0].unsqueeze(0)

                self.optimizer.zero_grad()
                out_grads = self.contact_net.phi_net.forward(prev_config)
                out_grads.backward(torch.ones_like(out_grads))

                self.optimizer.step()

