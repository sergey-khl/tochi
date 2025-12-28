import warp as wp
import warp.sim
import warp.sim.render
import numpy as np

# see https://github.com/DAIRLab/contact-nets/blob/main/README.md
# METER_SCALE = 0.0524 # this is half box width
METER_SCALE = 1 # this is half box width

# see https://github.com/DAIRLab/contact-nets/blob/main/data/params_processed/experiment.json
# To get the above ^^^ use these values
KE = 1e4
KD = 1e3
KF = 50.0
MU = 0.18
RESTITUTION = 0.125
SAMPLING_DT = 0.006756756756756757
MASS = 0.37

class Environment:
    def __init__(self, device, enable_timers=True):
        self.device = device
        self.up_axis = "Y"

        self.sim_substeps: int = 10 # featherstone
        self.sim_step = 0
        self.sim_time = 0.0
        self.sim_dt = SAMPLING_DT

        self.enable_timers = enable_timers

        self.model = self.init_model()
        self.integrator = self.init_integrator()
        self.renderer = self.init_renderer()

    """
    cpu use USDRenderer and GPU use OpenGL
    """
    def init_renderer(self):
        if self.device == "cpu":
            return wp.sim.render.SimRendererUsd(
                self.model,
                "cube_toss.usd",
                # up_axis=self.up_axis,
                # show_rigid_contact_points=True,
                # contact_points_radius=1e-3,
                # show_joints=True,
            )
        else:
            return wp.sim.render.SimRendererOpenGL(
                self.model,
                "cube toss",
                up_axis=self.up_axis,
                show_rigid_contact_points=True,
                contact_points_radius=1e-3,
                show_joints=True,
            )

    """
    this is the thing that actually solves the dynamics
    """
    def init_integrator(self):
        if self.device == "cpu":
            # return wp.sim.XPBDIntegrator()
            return wp.sim.FeatherstoneIntegrator(
                self.model, {
                    "update_mass_matrix_every": self.sim_substeps
                }
            )
        else:
            return wp.sim.FeatherstoneIntegrator(
                self.model, {
                    "update_mass_matrix_every": self.sim_substeps
                }
            )

    """
    setup of the environment. adds the cube and table.
    """
    def init_model(self):
        builder = wp.sim.ModelBuilder()

        b = builder.add_body(name="cube")

        box_width = METER_SCALE * 2.0
        volume = box_width ** 3
        density = MASS / volume

        builder.add_shape_box(
            b,
            hx=METER_SCALE,
            hy=METER_SCALE,
            hz=METER_SCALE,
            density=density,
            ke=KE,
            kd=KD,
            kf=KF,
            mu=MU,
        )

        builder.add_joint_free(child=b, parent=-1)
        builder.joint_q[2] = METER_SCALE * 1.5

        builder.set_ground_plane(
            ke=KE,
            kd=KD,
            kf=KF,
            mu=MU,
            restitution=RESTITUTION
        )

        model = builder.finalize(self.device)

        # intertia is whack but see in tests if its acutaly that bad
        
        self.dof_q_per_env = builder.joint_coord_count
        self.dof_qd_per_env = builder.joint_dof_count
        self.state_dim = self.dof_q_per_env + self.dof_qd_per_env

        return model

    def step(
        self,
        state: wp.sim.State,
        next_state: wp.sim.State,
        control: wp.sim.Control,
    ):
        self.extras = {}
        state.clear_forces()
        with wp.ScopedTimer(
            "collision_handling", color="orange", active=self.enable_timers
        ):
            wp.sim.collide(
                self.model,
                state,
                edge_sdf_iter=10,
                iterate_mesh_vertices=True,
            )
        print(self.integrator)

        with wp.ScopedTimer("simulation", color="red", active=self.enable_timers):
            self.integrator.simulate(
                self.model, state, next_state, self.sim_dt, control
            )
        self.sim_time += self.sim_dt
        self.sim_step += 1


    """
    render dynamics with opengl
    """
    def render(self):
        if self.renderer is None:
            raise Exception("need renderer to render")

        with wp.ScopedTimer("render", color="yellow", active=self.enable_timers):
            self.renderer.begin_frame(self.sim_time)
            render_state = (self.model.state())

            wp.sim.eval_fk(
                self.model,
                render_state.joint_q,
                render_state.joint_qd,
                None,
                render_state,
            )
            self.renderer.render(render_state)
            self.renderer.end_frame()

                
    """
    cleanup
    """
    def stop(self,):
        if self.device == "cpu":
            self.renderer.save()
