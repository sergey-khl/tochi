import warp as wp
import numpy as np
import math
import torch

# TODO: cite contact
class Trainer:
    def __init__(self):
        # TODO: look at G_bases shape but probably don't need a lot of this
        self.vertices = 8
        self.cone_bases = 8
        self.G_bases = self.compute_G_bases(self.cone_bases)
        self.G = self.compute_G()

        torch.set_printoptions(profile="full")
        print(self.G)
        print(self.G.shape)


    """
    alrighty.  TODO

    step 1: need state (curr state/control and next state)
    step 2: curr and next phi
    step 3: curr and next jacobian (both normal and tangential)
    step 4: curr gamma
    step 5: curr and next mass matrix
    step 6: curr and next force vectors
    step 7: pad for normal, tangent and slack
    ...
    mathematical pain

    """
    def compute_loss(self):
        # step 1
        pass
        


    def compute_G_bases(self, bases_n):
        bases = torch.zeros(bases_n, 2)
        for i, angle in enumerate(np.linspace(0, 2 * math.pi * (1 - 1 / bases_n), bases_n)):
            bases[i, 0] = math.cos(angle)
            bases[i, 1] = math.sin(angle)

        return bases

    def compute_G(self):
        """
        converts the bases into a block diagonal of shape (1, 16, 64)
        """
        block_diag = torch.block_diag(*[self.G_bases.T for _ in range(8)])
        return block_diag.unsqueeze(0)


