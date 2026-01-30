import warp as wp
import torch
from torch.nn import Module


@wp.func
def transform_point(
    pos: wp.vec3, 
    quat: wp.quat, 
    point_body: wp.vec3
):
    # Rotate point by quaternion and add translation
    return wp.quat_rotate(quat, point_body) + pos

@wp.func
def project_on_axis(
    point_world: wp.vec3, 
    axis: wp.vec3
):
    return wp.dot(point_world, axis)

@wp.kernel
def transform_and_project_kernel(
    # inputs
    configurations: wp.array(dtype=float, ndim=2),
    points: wp.array(dtype=wp.vec3),
    projections: wp.array(dtype=wp.vec3),
    # Outputs
    phi_out: wp.array(dtype=float, ndim=2)
):
    tid = wp.tid()
    
    p = wp.vec3(configurations[tid, 0], configurations[tid, 1], configurations[tid, 2])
    q = wp.quaternion(configurations[tid, 3], configurations[tid, 4], configurations[tid, 5], configurations[tid, 6])
    
    num_vertices = points.shape[0]
    num_projections = projections.shape[0]

    for v_idx in range(num_vertices):
        v_body = points[v_idx]
        v_world = transform_point(p, q, v_body)

        # Inner loop for projections
        for p_idx in range(num_projections):
            dist = project_on_axis(v_world, projections[p_idx])
            
            # Calculate flat index for output
            out_idx = v_idx * num_projections + p_idx 
            phi_out[tid, out_idx] = dist

# I think this will be used to store torch param info
class TransformAndProject3D(Module):
    def __init__(self, projections, points):
        super().__init__()
        self.projections = projections
        self.points = points

    # NOTE: in this case configurations is just one configuration
    def forward(self, configurations: torch.Tensor):
        return WarpWrapper.apply(
            configurations, 
            self.points, 
            self.projections, 
        )

class WarpWrapper(torch.autograd.Function):
    @staticmethod
    def forward(ctx, configurations, points, projections):
        # Allocate output
        ctx.batch_n = configurations.shape[0]
        num_contacts = points.shape[0]
        out_torch = torch.zeros((ctx.batch_n, num_contacts), device=configurations.device)
        ctx.phi_out = wp.from_torch(out_torch)

        # allocate input
        ctx.configurations = wp.from_torch(configurations)
        ctx.points_wp = wp.from_torch(points, dtype=wp.vec3, requires_grad=True)
        ctx.projections_wp = wp.from_torch(projections, dtype=wp.vec3, requires_grad=True)
        
        wp.launch(
            kernel=transform_and_project_kernel,
            dim=ctx.batch_n,
            inputs=[ctx.configurations, ctx.points_wp, ctx.projections_wp],
            outputs=[ctx.phi_out]
        )
        return wp.to_torch(ctx.phi_out)

    @staticmethod
    def backward(ctx, phi_out):
        ctx.phi_out.grad = wp.from_torch(phi_out)

        wp.launch(
            kernel=transform_and_project_kernel,
            dim=ctx.batch_n,
            inputs=[ctx.configurations, ctx.points_wp, ctx.projections_wp],
            outputs=[ctx.phi_out],
            adj_inputs=[ctx.configurations.grad, ctx.points_wp.grad, ctx.projections_wp.grad],
            adj_outputs=[ctx.phi_out.grad],
            adjoint=True,
        )

        # return adjoint w.r.t. inputs
        # TODO: wrong
        return (wp.to_torch(ctx.points_wp.grad), wp.to_torch(ctx.projections_wp.grad), None)
