"""
I believe the right thing to do is to mix torch and warp so that 
warp contains the loss formulation and torch must match in model definition to what
we want to use in warp. The idea being that torch already has a bunch of nice dataloader
and optimizer stuff but warp we can leverage the fast autodifferentiation as we can make
the entire sim step in one gpu kernel (I think).
"""
import torch
from torch import Tensor
import torch.nn as nn

from src.project_3d import TransformAndProject3D
from src.block_config import load_config
import warp as wp

class ContactNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.params = load_config()
        self.mu = self.params.environment_config.mu
        self.vertices = self.params.environment_config.vertices
        self.ground_normal = self.params.environment_config.ground_normal
        self.ground_tangent = self.params.environment_config.ground_tangent
        self.polynoise = self.params.noise

        self.init_polytope_net()
        
    def init_polytope_net(self):
        # here we basically only learn where the corners are and what the orientation of the sides are
        # we do grads in warp so we set requires_grad to false
        vertices_normal = nn.Parameter(self.vertices, requires_grad=False).contiguous()
        vertices_tangent = nn.Parameter(self.vertices, requires_grad=False).contiguous()

        zvec = nn.Parameter(self.ground_normal, requires_grad=True).contiguous()
        xyvecs = nn.Parameter(self.mu * self.ground_tangent, requires_grad=True).contiguous()

        with torch.no_grad():
            vertices_normal.add_(torch.randn_like(vertices_normal) * self.polynoise)
            vertices_tangent.add_(torch.randn_like(vertices_tangent) * self.polynoise)

            zvec.add_(torch.randn_like(zvec) * self.polynoise)
            xyvecs.add_(torch.randn_like(xyvecs) * self.polynoise * self.mu)

        self.phi_net = TransformAndProject3D(zvec, vertices_normal)
        self.tangent_net = TransformAndProject3D(xyvecs, vertices_tangent)
