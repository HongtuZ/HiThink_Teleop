import yaml
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from dex_retargeting.retargeting_config import RetargetingConfig
from .motion_controller.pinocchio_motion_control import PinocchioMotionControl

@dataclass
class TargetPose:
    head: np.ndarray = None
    left_wrist: np.ndarray = None
    right_wrist: np.ndarray = None
    left_fingers: np.ndarray = None
    right_fingers: np.ndarray = None
    left_griper: np.ndarray = None
    right_gripper: np.ndarray = None

@dataclass
class RobotJoints:
    left_arm: np.ndarray = None
    right_arm: np.ndarray = None
    left_hand: np.ndarray = None
    right_hand: np.ndarray = None

class Robot:
    def __init__(self, config_path:str):
        self.config = yaml.safe_load(open(config_path), 'r')
        self.left_arm_controller = None
        self.right_arm_controller = None
        self.left_hand_controller = None
        self.right_hand_controller = None
        self.init_controllers(self.config)

    def init_controllers(self, config):
        l_arm_config = config.get('left_arm', None)
        r_arm_config = config.get('right_arm', None)
        l_hand_config = config.get('left_hand', None)
        r_hand_config = config.get('right_hand', None)
        if l_arm_config:
            self.left_arm_controller = PinocchioMotionControl('left_arm', l_arm_config)
        if r_arm_config:
            self.right_arm_controller = PinocchioMotionControl('right_arm', r_arm_config)
        if l_hand_config:
            self.left_hand_controller = RetargetingConfig.from_dict(l_hand_config).buid()
        if r_hand_config:
            self.right_hand_controller = RetargetingConfig.from_dict(r_hand_config).build()

    def step(self, target_pose: TargetPose):
        result = RobotJoints()
        if self.left_arm_controller:
            self.left_arm_controller.step(target_pose.left_wrist)
            result.left_arm = self.left_arm_controller.get_current_qpos()
        if self.right_arm_controller:
            self.right_arm_controller.step(target_pose.right_wrist)
            result.right_arm = self.right_arm_controller.get_current_qpos()
        if self.left_hand_controller:
            result.left_hand = self.left_hand_controller.retarget(target_pose.left_fingers)
        if self.right_hand_controller:
            result.right_hand = self.right_hand_controller.retarget(target_pose.right_fingers)
        return result

