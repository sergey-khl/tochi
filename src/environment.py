import warp as wp
import warp.sim
import warp.sim.render
import numpy as np
from warp.sim.collide import get_box_vertex

import torch

# see https://github.com/DAIRLab/contact-nets/blob/main/README.md
# METER_SCALE = 0.0524 # this is half box width
METER_SCALE = 1.0 # this is half box width

# see https://github.com/DAIRLab/contact-nets/blob/main/data/params_processed/experiment.json
# To get the above ^^^ use these values
KE = 1e4
KD = 1e3
KF = 50.0
MU = 0.18
RESTITUTION = 0.125 # in contact nets this is also hard coded to 0 in train.py. verify 
SAMPLING_DT = 0.006756756756756757
MASS = 0.37

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
    def __init__(self, device, initial_qs, initial_qds, enable_timers=True):
        self.device = device
        self.up_axis = "Y"

        self.initial_qs = initial_qs
        self.initial_qds = initial_qds

        self.sim_substeps: int = 10 # featherstone
        self.sim_step = 0
        self.sim_time = 0.0
        self.sim_dt = SAMPLING_DT / self.sim_substeps
        # self.sim_dt = SAMPLING_DT
        self.num_envs = 1

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
            )
        else:
            return wp.sim.render.SimRendererOpenGL(
                self.model,
                "cube toss",
                up_axis=self.up_axis,
                # show_rigid_contact_points=True,
                contact_points_radius=1e-3,
                # show_joints=True,
                # scaling=5,
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
        up_vector = np.zeros(3)
        up_vector["xyz".index(self.up_axis.lower())] = 1.0
        builder = wp.sim.ModelBuilder(up_vector=up_vector)

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
        self.state = model.state()
        self.next_state = model.state()
        self.control = model.control()

        return model

    def step(
        self,
    ):
        for _ in range(self.sim_substeps):
            self.state.clear_forces()
            with wp.ScopedTimer(
                "collision_handling", color="orange", active=self.enable_timers
            ):
                wp.sim.collide(
                    self.model,
                    self.state,
                    edge_sdf_iter=10,
                    iterate_mesh_vertices=True,
                )
                # num_contacts = self.model.rigid_contact_count.numpy()[0]
                # print(num_contacts)

            with wp.ScopedTimer("simulation", color="red", active=self.enable_timers):
                self.integrator.simulate(
                    self.model, self.state, self.next_state, self.sim_dt, self.control
                )
            self.sim_time += self.sim_dt
            self.sim_step += 1
            # set next state as current state
            self.state, self.next_state = self.next_state, self.state # TODO: understand this
            # self.state = self.next_state


    """
    render dynamics with opengl. dont use cpu its annoying
    """
    def render(self):
        if self.renderer is None:
            raise Exception("need renderer to render")

        with wp.ScopedTimer("render", color="yellow", active=self.enable_timers):
            self.renderer.begin_frame(self.sim_time)
            render_state = (self.state)
            print("render state")
            print(render_state.joint_q, render_state.joint_qd)

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
                cam_pos = wp.vec3(0., 3., 20.)
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
        qi = [0, 1, 2, 3, 4, 5, 6]
        qdi = [7, 8, 9, 10, 11, 12]
        q = torch_state[qi].float()
        qd = torch_state[qdi].float()
        x, quat = q[0:3], q[3:7]
        # ang_vel_body, lin_vel = qd[0:3], qd[3:6]
        # ang_vel_world = quat_rotate(quat.unsqueeze(0), ang_vel_body.unsqueeze(0)).squeeze(0)
        # lin_vel_world = lin_vel + torch.cross(x, ang_vel_world)
        # qd[0:3] = ang_vel_world
        # qd[3:6] = lin_vel_world
        return q, qd

    def set_torch_state(
        self,
        curr_torch_state,
        next_torch_state,
        eval_fk: bool = True,
    ):
        # format of the state in ContactNets is as follows:
        # position (3), quaternion (4), velocity (3), angular velocity (3), control (6)
        q, qd = self.torch_state_to_q_qd(curr_torch_state)
        self.state.joint_q.assign(wp.from_torch(q))
        self.state.joint_qd.assign(wp.from_torch(qd))
        q, qd = self.torch_state_to_q_qd(next_torch_state)
        self.next_state.joint_q.assign(wp.from_torch(q))
        self.next_state.joint_qd.assign(wp.from_torch(qd))
        # self.control.joint_act.assign(wp.array([1., 1., 1., 1., 1., 1., 1.]))
        # if eval_fk:
        wp.sim.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, None, self.state)


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

