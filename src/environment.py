import warp as wp
import warp.sim
import warp.sim.render
import numpy as np
from warp.sim.collide import get_box_vertex

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
RESTITUTION = 0.125 # in contact nets this is also hard coded to 0 in train.py. verify 
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
def generate_contact_pairs(
    geo: warp.sim.ModelShapeGeometry,
    shape_shape_collision: wp.array(dtype=bool),
    num_shapes_per_env: int,
    num_contacts_per_env: int,
    ground_shape_index: int,
    shape_body: wp.array(dtype=int),
    up_vector: wp.vec3,
    shape_X_bs: wp.array(dtype=wp.transform),
    # outputs
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_depth: wp.array(dtype=float),
    contact_thickness: wp.array(dtype=float),
):
    env_id = wp.tid()

    shape_offset = num_shapes_per_env * env_id
    contact_idx = num_contacts_per_env * env_id
    for i in range(num_shapes_per_env):
        body = shape_body[shape_offset + i]

        if body == -1:
            # static shapes are ignored, e.g. ground
            continue
        
        if shape_shape_collision[shape_offset + i] == False:
            # filter out visual meshes
            continue

        geo_type = geo.type[shape_offset + i]
        geo_scale = geo.scale[shape_offset + i]
        geo_thickness = geo.thickness[shape_offset + i]
        shape_tf = shape_X_bs[shape_offset + i]

        if geo_type == wp.sim.GEO_SPHERE:
            contact_shape0[contact_idx] = shape_offset + i
            contact_shape1[contact_idx] = ground_shape_index
            contact_point0[contact_idx] = wp.transform_get_translation(shape_tf)
            contact_point1[contact_idx] = wp.vec3(0.0)
            contact_normal[contact_idx] = up_vector
            contact_depth[contact_idx] = 1000.0
            contact_thickness[contact_idx] = geo_scale[0]
            contact_idx += 1
        
        if geo_type == wp.sim.GEO_CAPSULE:
            # add points at the two ends of the capsule
            contact_shape0[contact_idx] = shape_offset + i
            contact_shape1[contact_idx] = ground_shape_index
            contact_point0[contact_idx] = wp.transform_point(
                shape_tf, wp.vec3(0.0, geo_scale[1], 0.0)
            )
            contact_point1[contact_idx] = wp.vec3(0.0)
            contact_normal[contact_idx] = up_vector
            contact_depth[contact_idx] = 1000.0
            contact_thickness[contact_idx] = geo_scale[0]
            contact_idx += 1
            
            contact_shape0[contact_idx] = shape_offset + i
            contact_shape1[contact_idx] = ground_shape_index
            contact_point0[contact_idx] = wp.transform_point(
                shape_tf, wp.vec3(0.0, -geo_scale[1], 0.0)
            )
            contact_point1[contact_idx] = wp.vec3(0.0)
            contact_normal[contact_idx] = up_vector
            contact_depth[contact_idx] = 1000.0
            contact_thickness[contact_idx] = geo_scale[0]
            contact_idx += 1

        if geo_type == wp.sim.GEO_BOX:
            # add box corner points
            for j in range(8):
                p = get_box_vertex(j, geo_scale)
                contact_shape0[contact_idx] = shape_offset + i
                contact_shape1[contact_idx] = ground_shape_index
                contact_point0[contact_idx] = wp.transform_point(shape_tf, p)
                contact_point1[contact_idx] = wp.vec3(0.0)
                contact_normal[contact_idx] = up_vector
                contact_depth[contact_idx] = 1000.0
                contact_thickness[contact_idx] = geo_thickness
                contact_idx += 1
        
        # TODO: temporary fix for mesh body
        if geo_type != wp.sim.GEO_BOX and geo_type != wp.sim.GEO_CAPSULE and geo_type != wp.sim.GEO_SPHERE:
            contact_shape0[contact_idx] = shape_offset + i
            contact_shape1[contact_idx] = ground_shape_index
            contact_point0[contact_idx] = wp.transform_point(
                shape_tf, wp.vec3(0.0, 0.0, 0.0)
            )
            contact_point1[contact_idx] = wp.vec3(0.0)
            contact_normal[contact_idx] = up_vector
            contact_depth[contact_idx] = 1000.0
            contact_thickness[contact_idx] = 0.0
            contact_idx += 1

@wp.kernel(enable_backward=False)
def collision_detection_ground(
    body_q: wp.array(dtype=wp.transform),
    shape_X_bs: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    geo: warp.sim.ModelShapeGeometry,
    ground_shape_index: int,
    contact_shape0: wp.array(dtype=int),
    contact_point0: wp.array(dtype=wp.vec3),
    # outputs
    contact_shape1: wp.array(dtype=int),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_depth: wp.array(dtype=float),
    contact_offset0: wp.array(dtype=wp.vec3),
    contact_offset1: wp.array(dtype=wp.vec3)
):
    contact_id = wp.tid()
    shape = contact_shape0[contact_id]
    ground_tf = shape_X_bs[ground_shape_index]
    body = shape_body[shape]
    point_world = wp.transform_point(body_q[body], contact_point0[contact_id])

    # contact shape1 is always ground
    contact_shape1[contact_id] = ground_shape_index
    
    # get contact normal in world frame
    ground_up_vec = wp.vec3(0., 1., 0.)
    contact_normal[contact_id] = wp.transform_vector(ground_tf, ground_up_vec)
    
    # transform point to ground shape frame
    T_world_to_ground = wp.transform_inverse(ground_tf)
    point_plane = wp.transform_point(T_world_to_ground, point_world)
    
    # get contact depth
    contact_depth[contact_id] = wp.dot(point_plane, ground_up_vec)
    
    # project to plane
    projected_point = point_plane - contact_depth[contact_id] * ground_up_vec
    
    # transform to world frame (applying the shape transform)
    contact_point1[contact_id] = wp.transform_point(ground_tf, projected_point)

    # compute contact offsets
    T_world_to_body0 = wp.transform_inverse(body_q[body])
    thickness_body0 = geo.thickness[shape]
    thickness_ground = geo.thickness[ground_shape_index]
    contact_offset0[contact_id] = wp.transform_vector(
        T_world_to_body0, 
        -thickness_body0 * contact_normal[contact_id]
    )
    contact_offset1[contact_id] = wp.transform_vector(
        T_world_to_ground,
        thickness_ground * contact_normal[contact_id]
    )

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
        self.sim_dt = SAMPLING_DT / 10
        self.num_envs = 1

        self.enable_timers = enable_timers

        self.model = self.init_model()
        self.integrator = self.init_integrator()
        self.renderer = self.init_renderer()
        # self.initialize_contacts(self.model)

    # Construct the ordered list of contact pairs
    def initialize_contacts(self, model: wp.sim.Model):
        # compute number of contact pairs per env
        # NOTE: only work for ground env for now
        num_shapes_per_env = (model.shape_count - 1) // model.num_envs
        num_contacts_per_env = 0
        geo_types = model.shape_geo.type.numpy()
        shape_body = model.shape_body.numpy()
        for i in range(num_shapes_per_env):
            # static shapes are ignored, e.g. ground
            if shape_body[i] == -1:
                continue
            # filter out visual meshes
            if model.shape_shape_collision[i] == False:
                continue
            
            geo_type = geo_types[i]
            if geo_type == wp.sim.GEO_SPHERE:
                num_contacts_per_env += 1
            elif geo_type == wp.sim.GEO_CAPSULE:
                num_contacts_per_env += 2
            elif geo_type == wp.sim.GEO_BOX:
                num_contacts_per_env += 8
            else: # TODO: temporary fix for mesh body
                num_contacts_per_env += 1
        
        self.abstract_contacts = AbstractContact(
            num_contacts_per_env = num_contacts_per_env,
            num_envs = model.num_envs,
            model = model, 
            device = warp_utils.device_to_torch(model.device)
        )

        # Generate contact points once at the beginning of the simulation
        wp.launch(
            generate_contact_pairs,
            dim=self.model.num_envs,
            inputs=[
                model.shape_geo,
                wp.from_numpy(np.array(model.shape_shape_collision, dtype=bool)),
                num_shapes_per_env,
                num_contacts_per_env,
                model.shape_count - 1,  # ground plane index
                model.shape_body,
                model.up_vector,
                model.shape_transform,
            ],
            outputs=[
                model.rigid_contact_shape0,
                model.rigid_contact_shape1,
                model.rigid_contact_point0,
                model.rigid_contact_point1,
                model.rigid_contact_normal,
                model.rigid_contact_depth,
                model.rigid_contact_thickness,
            ],
            device=model.device,
        )

        model.rigid_contact_max = self.abstract_contacts.num_total_contacts
        model.rigid_contact_count = wp.array(
            [model.rigid_contact_max], 
            dtype=wp.int32, 
            device=model.device
        )


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
            pos=wp.vec3(0., METER_SCALE * 1.5, 0.),
            rot=wp.quat(0., 0., 0., 1.),
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

        # wp.launch(
        #     collision_detection_ground,
        #     dim=self.model.rigid_contact_max,
        #     inputs=[
        #         self.state.body_q,
        #         self.model.shape_transform,
        #         self.model.shape_body,
        #         self.model.shape_geo,
        #         self.model.shape_count - 1,  # ground plane index
        #         self.model.rigid_contact_shape0,
        #         self.model.rigid_contact_point0,
        #     ],
        #     outputs=[
        #         self.model.rigid_contact_shape1,
        #         self.model.rigid_contact_point1,
        #         self.model.rigid_contact_normal,
        #         self.model.rigid_contact_depth,
        #         self.model.rigid_contact_offset0,
        #         self.model.rigid_contact_offset1
        #     ],
        #     device=self.device,
        # )
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
                num_contacts = self.model.rigid_contact_count.numpy()[0]
                print(num_contacts)

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

