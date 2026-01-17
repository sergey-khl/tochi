import warp as wp

class Project3D:
    def __init__(self, projections, points):
        self.projections = projections
        self.points = points

    def transformAndProject(self):
        wp.launch(
            kernel=transform_and_project_kernel,
            dim=batch_size,
            inputs=[configs_wp, verts_wp, proj_wp, phi_wp]
        )


@wp.func
def transform_point(
    pos: wp.vec3, 
    quat: wp.vec4, 
    point_body: wp.vec3
):
    # Rotate point by quaternion and add translation
    # Warp has built-in quaternion functions
    return wp.quat_rotate(quat, point_body) + pos

@wp.func
def project_on_axis(
    point_world: wp.vec3, 
    axis: wp.vec3
):
    return wp.dot(point_world, axis)

# --- 2. The Fused Kernel (Forward Pass) ---

@wp.kernel
def transform_and_project_kernel(
    # Inputs (Batch x 7) -> [px, py, pz, qx, qy, qz, qw]
    configs: wp.array(dtype=float, ndim=2),
    # Shape Definition (Body Frame)
    vertices: wp.array(dtype=wp.vec3),
    projections: wp.array(dtype=wp.vec3),
    # Outputs
    phi_out: wp.array(dtype=float, ndim=2) # Batch x (Num_Projections)
):
    tid = wp.tid()
    
    # 1. Unpack Configuration (Row 'tid')
    # Assuming config is [px, py, pz, qx, qy, qz, qw]
    p = wp.vec3(configs[tid, 0], configs[tid, 1], configs[tid, 2])
    q = wp.vec4(configs[tid, 3], configs[tid, 4], configs[tid, 5], configs[tid, 6])
    
    # 2. Iterate over the geometry
    # In the python code, it implies projections correspond to points 1-to-1 
    # or are interleaved. Let's assume 1-to-1 for simplicity here, 
    # but you can add nested loops if it's all-pairs.
    
    num_contacts = vertices.shape[0]
    
    for i in range(num_contacts):
        v_body = vertices[i]
        n_dir = projections[i]
        
        # Transform
        v_world = transform_point(p, q, v_body)
        
        # Project
        # Result = distance along the projection vector
        phi_out[tid, i] = project_on_axis(v_world, n_dir)
