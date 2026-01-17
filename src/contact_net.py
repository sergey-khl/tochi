"""
I believe the right thing to do is to mix torch and warp so that 
warp contains the loss formulation and torch must match in model definition to what
we want to use in warp. The idea being that torch already has a bunch of nice dataloader
and optimizer stuff but warp we can leverage the fast autodifferentiation as we can make
the entire sim step in one gpu kernel (I think).
"""
import torch
from torch.nn as nn
import warp as wp

class ContactNet(nn.Module):
    def __init__(self, mu, polynoise = 0.4):
        super().__init__()

        self.mu = mu
        self.polynoise = polynoise
        
    # these are the ground truth that we should come to near the end of training
    def corners(self):
        return torch.tensor([[-1, -1, -1, -1, 1, 1, 1, 1],
                             [-1, -1, 1, 1, -1, -1, 1, 1],
                             [1, -1, -1, 1, 1, -1, -1, 1]]).t().double()
    def ground_normal(self): return torch.tensor([[0.0, 0, 1]])
    def ground_tangent(self): return torch.tensor([[1.0, 0, 0], [0, 1, 0]])

    def init_polytope_net(self):
        vertices_normal = nn.Parameter(self.corners(), requires_grad=True)
        vertices_tangent = nn.Parameter(self.corners(), requires_grad=True)

        zvec = nn.Parameter(self.ground_normal(), requires_grad=True)
        xyvecs = nn.Parameter(self.mu.item() * self.ground_tangent(), requires_grad=True)

        with torch.no_grad():
            vertices_normal.add_(torch.randn(vertices_normal.size()) * self.polynoise)
            vertices_tangent.add_(torch.randn(vertices_tangent.size()) * self.polynoise)

            zvec.add_(torch.randn(zvec.size()) * self.polynoise)
            xyvecs.add_(torch.randn(xyvecs.size()) * self.polynoise * self.mu.item())

        self.phi_net = TransformAndProject3D(zvec, vertices_normal)
        self.tangent_net = TransformAndProject3D(xyvecs, vertices_tangent)

        
    def _run_warp_op(self, configs_torch, compute_grad=False):
        """
        Helper to bridge PyTorch -> Warp -> PyTorch
        """
        batch_size = configs_torch.shape[0]
        num_contacts = self.vertices_body.shape[0]

        # 1. Zero-Copy Transfer to Warp
        #    Use requires_grad=True if we want to differentiate w.r.t inputs (Jacobian)
        configs_wp = wp.from_torch(configs_torch, requires_grad=compute_grad)
        verts_wp = wp.from_torch(self.vertices_body)
        proj_wp = wp.from_torch(self.projections)
        
        # Output buffer
        phi_wp = wp.zeros((batch_size, num_contacts), dtype=float, requires_grad=compute_grad)
        
        # 2. Run Kernel / Tape
        tape = wp.Tape() if compute_grad else None
        
        # If we need gradients, we record on the tape
        if compute_grad:
            with tape:
                wp.launch(
                    kernel=transform_and_project_kernel,
                    dim=batch_size,
                    inputs=[configs_wp, verts_wp, proj_wp, phi_wp]
                )
        else:
            
        return phi_wp, configs_wp, tape

    def compute_phi(self, configurations: torch.Tensor) -> torch.Tensor:
        """
        Replaces the old forward pass.
        Returns: phi (Batch x Contact_N)
        """
        # Run Warp forward
        phi_wp, _, _ = self._run_warp_op(configurations, compute_grad=False)
        
        # Return as PyTorch tensor (Zero Copy)
        return wp.to_torch(phi_wp)

    def compute_Jn(self, configurations: torch.Tensor) -> torch.Tensor:
        """
        Replaces JacProp! 
        Computes Jacobian d(phi)/d(config) using Warp AD.
        """
        batch_size = configurations.shape[0]
        num_contacts = self.vertices_body.shape[0]
        input_dim = 7 # Pose dimension
        
        # 1. Run Forward with Tape
        phi_wp, config_wp, tape = self._run_warp_op(configurations, compute_grad=True)
        
        # 2. Extract Jacobian Row-by-Row
        #    We need J of shape (Batch, Contact_N, 7)
        
        jac_rows = []
        
        # We iterate over the *output* dimension (Number of contacts)
        # This is standard for Reverse-Mode AD
        for i in range(num_contacts):
            tape.zero()
            
            # Select the i-th contact for every batch item
            grad_output = torch.zeros((batch_size, num_contacts), device=configurations.device)
            grad_output[:, i] = 1.0 # One-hot vector
            
            # Pass gradient into Warp
            phi_wp.grad = wp.from_torch(grad_output)
            
            # Backward pass
            tape.backward()
            
            # The result is in config_wp.grad
            # This is d(phi_i) / d(config)
            grad_in = wp.to_torch(config_wp.grad).clone() # Clone is important!
            jac_rows.append(grad_in)
            
        # Stack to get (Batch, Contact_N, 7)
        # Permute to match expected shape if necessary
        jacobian = torch.stack(jac_rows, dim=1)
        
        return jacobian
