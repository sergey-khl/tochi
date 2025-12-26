import warp as wp
import warp.sim


class Environment:
    def __init__(self, device):
        self.device = device

        self.model = self.init_model()

    """
    setup of the environment. adds the cube and table.
    """
    def init_model(self):
        builder = wp.sim.ModelBuilder()

        builder.add_shape_plane(
            pos=(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            normal=(0.0, 1.0, 0.0)
        )

        builder.add_body_box(
            pos=(0.0, 2.0, 0.0),       # Start 2 meters up
            rot=wp.quat_from_axis_angle((1.0, 0.0, 0.0), 0.5), # Tilted slightly
            hx=0.5, hy=0.5, hz=0.5,    # Half-extents (dimensions)
            mass=1.0,
            density=100.0,
            name="falling_box"
        )

        model = builder.finalize(self.device)
        
        return model
