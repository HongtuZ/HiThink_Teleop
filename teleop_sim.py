from isaacgym import gymapi
from isaacgym import gymtorch
import torch
import math
import numpy as np
from pathlib import Path
from copy import deepcopy
from typing import Dict, Any

from avp_stream.utils.isaac_utils import *
from avp_stream.utils.se3_utils import *
from avp_stream.utils.trn_constants import *

from robot.motion_controller.pinocchio_motion_control import PinocchioMotionControl
import time
import common.helper as helper
from omegaconf import OmegaConf

def np2tensor(
    data: Dict[str, np.ndarray], device: torch.device
) -> Dict[str, torch.Tensor]:
    return {
        key: torch.tensor(value, dtype=torch.float32, device=device)
        for key, value in data.items()
    }

class RealRobot:
    def __init__(self, config, init_joints):
        self.motion_controller = PinocchioMotionControl(
            urdf_path=config.urdf_path,
            ee_name=config.ee_name,
            ik_dt=config.ik_dt,
            ik_damping=config.ik_damping,
            ik_eps=config.ik_eps,
        )
        self.init_joints = init_joints
        self.init_ee_pose = self.motion_controller.compute_ee_pose(init_joints)
        self.old_ee_pose = self.init_ee_pose
        self.init_cmd = None

    def step(self, cmd):
        if self.init_cmd is None:
            self.init_cmd = cmd.copy()
        cmd[:3] = cmd[:3] - self.init_cmd[:3] + self.init_ee_pose[:3]
        cmd[:7] = helper.smooth_pose(self.old_ee_pose[:7], cmd[:7], alpha=0.5)
        self.old_ee_pose = cmd.copy()
        print(f'cmd: {cmd}')
        print(f'init cmd: {self.init_cmd}')
        print(f'init pose: {self.init_ee_pose}')
        target_pose = cmd[:7]
        return target_pose, self.motion_controller.step(cmd[:7], repeat=10)

    def set_current_joints(self, joints):
        self.motion_controller.set_current_qpos(joints)



class Sim:
    def __init__(self, config) -> None:
        self.num_envs = 1

        # initialize gym
        self.gym = gymapi.acquire_gym()

        # configure sim
        self.device = "cpu"
        self.sim_params = default_sim_params(use_gpu=(self.device == "cuda:0"))

        # create sim
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, self.sim_params)
        if self.sim is None:
            raise Exception("Failed to create sim")

        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.distance = 0.0
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

        # create environment
        num_envs = 1
        num_per_row = int(math.sqrt(num_envs))
        env_spacing = 1.25
        env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, 2*env_spacing)

        np.random.seed(17)

        self.env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)

        # add robot asset
        robot_asset_root = Path(config.urdf_path).parent
        robot_asset_file = Path(config.urdf_path).name
        asset_options = self.get_asset_options()

        # load robot asset
        robot_asset = self.gym.load_asset(
            self.sim, str(robot_asset_root), robot_asset_file, asset_options
        )
        self.dof = self.gym.get_asset_dof_count(robot_asset)
        self.robot_dof_props = self.gym.get_asset_dof_properties(robot_asset)
        self.set_dof_properties()

        # create robot and place it
        self.left_arm_base_pos = np.array([0, 0.2, 1.1])
        self.right_arm_base_pos = np.array([0, -0.2, 1.1])
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0, 0.2, 1.1)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        self.left_arm_handle = self.create_robot('left_arm', robot_asset, pose)
        pose.p = gymapi.Vec3(0, -0.2, 1.1)
        self.right_arm_handle = self.create_robot('right_arm', robot_asset, pose)

        # create teleop wrist axis
        self.axis = self.load_axis('normal')
        self.left_wrist_axis = self.gym.create_actor(self.env, self.axis, gymapi.Transform(), 'left_wrist', 1)
        self.right_wrist_axis = self.gym.create_actor(self.env, self.axis, gymapi.Transform(), 'right_wrist', 2)

        # create real robot controller
        left_arm_init_joints = self.gym.get_actor_dof_states(self.env, self.left_arm_handle, gymapi.STATE_POS)['pos']
        self.left_arm_robot = RealRobot(config, left_arm_init_joints)
        right_arm_init_joints = self.gym.get_actor_dof_states(self.env, self.right_arm_handle, gymapi.STATE_POS)['pos']
        self.right_arm_robot = RealRobot(config, right_arm_init_joints)

        # create default viewer
        self.create_viewer()
        self.gym.prepare_sim(self.sim)
        self.initialize_tensors()


    def get_asset_options(self) -> gymapi.AssetOptions:
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

        return asset_options

    def set_dof_properties(self) -> None:
        for i in range(self.dof):
            self.robot_dof_props["stiffness"][i] = 1000.0
            self.robot_dof_props["damping"][i] = 1000.0

    def create_robot(self, name: str, robot_asset: Any, pose: gymapi.Transform) -> None:
        robot_handle = self.gym.create_actor(
            self.env, robot_asset, pose, name
        )
        self.gym.set_actor_dof_properties(
            self.env, robot_handle, self.robot_dof_props
        )
        self.gym.set_actor_dof_states(
            self.env,
            robot_handle,
            np.zeros(self.dof, gymapi.DofState.dtype),
            gymapi.STATE_ALL,
        )
        self.gym.set_actor_dof_position_targets(
            self.env, robot_handle, np.zeros(self.dof, dtype=np.float32)
        )
        return robot_handle

    def load_axis(self, size):
        urdf_root = Path(__file__).parent / 'assets/urdf/axis'
        robot_asset_file = f'{size}_axis.urdf'
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        robot_asset = self.gym.load_asset(self.sim, str(urdf_root), robot_asset_file, asset_options)
        return robot_asset

    def create_viewer(self) -> None:
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            raise Exception("Failed to create viewer")

        cam_pos = gymapi.Vec3(-1, 0, 2)
        cam_target = gymapi.Vec3(0, 0, 1)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def refresh_tensors(self): 
        # refresh tensors
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

    def initialize_tensors(self): 
        self.refresh_tensors()
        # get jacobian tensor
        # get rigid body state tensor
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_states).view(self.num_envs, -1, 13)

        # get actor root state tensor
        _root_states = self.gym.acquire_actor_root_state_tensor(self.sim)
        root_states = gymtorch.wrap_tensor(_root_states).view(self.num_envs, -1, 13)
        self.root_state = root_states
        print(self.root_state.shape)
        print(self.root_state)

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, False)
        self.gym.sync_frame_time(self.sim)

    def modify_root_state(self, l_pose, r_pose): 
        new_root_state = self.root_state
        new_root_state = deepcopy(self.root_state)
        new_root_state[:, 2, :7] = torch.tensor(l_pose)
        new_root_state[:, 3, :7] = torch.tensor(r_pose)
        new_root_state = new_root_state.view(-1, 13)

        return new_root_state

    def step(
        self, control_cmd: dict
    ) -> None:
        left_arm_states = self.gym.get_actor_dof_states(self.env, self.left_arm_handle, gymapi.STATE_POS)
        self.left_arm_robot.set_current_joints(left_arm_states['pos'])
        right_arm_states = self.gym.get_actor_dof_states(self.env, self.right_arm_handle, gymapi.STATE_POS)
        self.right_arm_robot.set_current_joints(right_arm_states['pos'])

        left_target_pose, (left_ik_success, left_arm_target_joints) = self.left_arm_robot.step(control_cmd['left_arm'])
        right_target_pose, (right_ik_success, right_arm_target_joints) = self.right_arm_robot.step(control_cmd['right_arm'])
        # Set robot DOF
        print(f'left joints:', [f'{v:.2f}' for v in left_arm_target_joints])
        print(f'righ joints:', [f'{v:.2f}' for v in right_arm_target_joints])
        states = np.zeros(self.dof, dtype=gymapi.DofState.dtype)
        if left_ik_success:
            states["pos"] = left_arm_target_joints
            self.gym.set_actor_dof_states(
                self.env, self.left_arm_handle, states, gymapi.STATE_POS
            )
        if right_ik_success:
            states["pos"] = right_arm_target_joints
            self.gym.set_actor_dof_states(
                self.env, self.right_arm_handle, states, gymapi.STATE_POS
            )
        # Step the physics
        self.gym.simulate(self.sim)
        self.refresh_tensors()

        # set axis
        left_target_pose[:3] += self.left_arm_base_pos
        right_target_pose[:3] += self.right_arm_base_pos
        new_root_state = self.modify_root_state(left_target_pose, right_target_pose)
        env_side_actor_idxs = torch.arange(0, 4, dtype = torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(new_root_state), gymtorch.unwrap_tensor(env_side_actor_idxs), len(env_side_actor_idxs))

        # render
        self.gym.fetch_results(self.sim, True)
        self.gym.render_all_camera_sensors(self.sim)

        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)
        time.sleep(0.05)


    def end(self) -> None:
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

if __name__ == '__main__':
    config_path = Path(__file__).parent/'config/hithink_robot.yaml'
    config = OmegaConf.load(config_path)
    sim = Sim(config.left_arm.config)
    while True:
        robot_cmd = {}
        robot_cmd['left_arm'] = np.array([0.2,0.2,0.0,0,0,0,1])
        robot_cmd['right_arm'] = np.array([0.1,0,0.5,0,0,0,1])
        sim.step(robot_cmd)