import warp as wp
import warp.sim
import warp.sim.render
import numpy as np

import torch

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

# TODO: cite nerd
@wp.func
def has_ground_penetration(pos: wp.vec3, quat: wp.quat):
    margin = 1e-3
    tf = wp.transform(pos, quat)
    if wp.transform_point(tf, wp.vec3(METER_SCALE, METER_SCALE, METER_SCALE))[2] < margin:
        return True
    if wp.transform_point(tf, wp.vec3(-METER_SCALE, METER_SCALE, METER_SCALE))[2] < margin:
        return True
    if wp.transform_point(tf, wp.vec3(METER_SCALE, -METER_SCALE, METER_SCALE))[2] < margin:
        return True
    if (wp.transform_point(tf, wp.vec3(-METER_SCALE, -METER_SCALE, METER_SCALE))[2] < margin):
        return True
    if wp.transform_point(tf, wp.vec3(METER_SCALE, METER_SCALE, -METER_SCALE))[2] < margin:
        return True
    if (wp.transform_point(tf, wp.vec3(-METER_SCALE, METER_SCALE, -METER_SCALE))[2] < margin):
        return True
    if (wp.transform_point(tf, wp.vec3(METER_SCALE, -METER_SCALE, -METER_SCALE))[2] < margin):
        return True
    if (wp.transform_point(tf, wp.vec3(-METER_SCALE, -METER_SCALE, -METER_SCALE))[2] < margin):
        return True
    return False

@wp.kernel(enable_backward=False)
def reset_cube(
    dof_q_per_env: int,
    dof_qd_per_env: int,
    default_joint_q_init: wp.array(dtype=wp.float32),
    default_joint_qd_init: wp.array(dtype=wp.float32),
    # outputs
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
):
    env_id = wp.tid()

    for i in range(dof_q_per_env):
        joint_q[env_id * dof_q_per_env + i] = default_joint_q_init[
            env_id * dof_q_per_env + i
        ]
    for i in range(dof_qd_per_env):
        joint_qd[env_id * dof_qd_per_env + i] = default_joint_qd_init[
            env_id * dof_qd_per_env + i
        ]


@torch.jit.script
def quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c

class Environment:
    def __init__(self, device, enable_timers=True):
        self.device = device
        self.up_axis = "Y"

        self.initial_qs = None
        self.initial_qds = None

        self.sim_substeps: int = 10 # featherstone
        self.sim_step = 0
        self.sim_time = 0.0
        self.sim_dt = SAMPLING_DT
        self.num_envs = 1

        self.enable_timers = enable_timers

        self.model = self.init_model()
        self.integrator = self.init_integrator()
        self.renderer = self.init_renderer()

        self.state = self.model.state()

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
                scaling=5,
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
        builder.joint_q[1] = METER_SCALE * 1.5

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

            # update cam position
            if self.device != "cpu":
                # pass
                cam_pos = wp.vec3(0., 3., 50.)
                with wp.ScopedTimer("update_view_matrix", color=0x663300, active=self.enable_timers):
                    self.renderer.update_view_matrix(cam_pos=cam_pos)


            self.renderer.render(render_state)
            self.renderer.end_frame()

                
    """
    cleanup
    """
    def stop(self,):
        if self.device == "cpu":
            self.renderer.save()

    @staticmethod
    def torch_state_to_q_qd(torch_state):
        # format of ContactNets dataset:
        # position (3), quaternion (4), velocity (3), angular velocity (3), control (6)
        qi = [0, 1, 2, 4, 5, 6, 3]
        qdi = [10, 11, 12, 7, 8, 9]
        if torch_state.dim() == 1:
            q = torch_state[qi].float()
            qd = torch_state[qdi].float()
            x, quat = q[0:3], q[3:7]
            ang_vel_body, lin_vel = qd[0:3], qd[3:6]
            ang_vel_world = quat_rotate(quat.unsqueeze(0), ang_vel_body.unsqueeze(0)).squeeze(0)
            lin_vel_world = lin_vel + torch.cross(x, ang_vel_world)
            qd[0:3] = ang_vel_world
            qd[3:6] = lin_vel_world
        else:
            q = torch_state[:, qi].float()
            qd = torch_state[:, qdi].float()

            if True:
                x, quat = q[:, 0:3], q[:, 3:7]
                ang_vel_body, lin_vel = qd[:, 0:3], qd[:, 3:6]
                ang_vel_world = quat_rotate(quat, ang_vel_body)
                lin_vel_world = lin_vel + torch.cross(x, ang_vel_world, dim = -1)
                qd[:, 0:3] = ang_vel_world
                qd[:, 3:6] = lin_vel_world
        return q, qd

    def set_torch_state(
        self,
        torch_state,
        eval_fk: bool = True,
    ):
        # format of the state in ContactNets is as follows:
        # position (3), quaternion (4), velocity (3), angular velocity (3), control (6)
        q, qd = self.torch_state_to_q_qd(torch_state)
        self.state.joint_q.assign(wp.from_torch(q))
        self.state.joint_qd.assign(wp.from_torch(qd))
        # if eval_fk:
        #     wp.sim.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, None, state)


    def reset_envs(self, env_ids: wp.array = None):
        """Reset environments where env_ids buffer indicates True. Resets all envs if env_ids is None."""
        wp.launch(
            reset_cube,
            dim=self.num_envs,
            inputs=[
                env_ids,
                self.dof_q_per_env,
                self.dof_qd_per_env,
                self.model.joint_q,
                self.model.joint_qd,
            ],
            outputs=[
                self.state.joint_q,
                self.state.joint_qd,
            ],
            device=self.device,
        )
        self.seed += self.num_envs

