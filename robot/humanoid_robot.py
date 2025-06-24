from omegaconf import OmegaConf
from typing import Dict
from .robot_components import RobotComponents
import numpy as np

class HumanoidRobot:

    def __init__(self, config_path):
        config = OmegaConf.load(config_path)
        self.head = RobotComponents[config.head.type]()
        self.left_arm = RobotComponents[config.left_arm.type](can_id=config.left_arm.can_id, is_left=True)
        self.right_arm = RobotComponents[config.right_arm.type](can_id=config.right_arm.can_id, is_left=False)
        self.left_hand = RobotComponents[config.left_hand.type]()
        self.right_hand = RobotComponents[config.right_hand.type]()
        self.left_leg = RobotComponents[config.left_leg.type]()
        self.right_leg = RobotComponents[config.right_leg.type]()
        self.torso = RobotComponents[config.torso.type]()

    def step(self, control_cmd: Dict):
        """
        Update the robot's state based on control commands.
        :param control_cmd: A dictionary containing control commands for different body parts.
        """
        if self.head and 'head' in control_cmd:
            self.head.step(control_cmd['head'])
        if self.left_arm and 'left_arm' in control_cmd:
            self.left_arm.step(control_cmd['left_arm'])
        if self.right_arm and 'right_arm' in control_cmd:
            self.right_arm.step(control_cmd['right_arm'])
        if self.left_hand and 'left_hand' in control_cmd:
            self.left_hand.step(control_cmd['left_hand'])
        if self.right_hand and 'right_hand' in control_cmd:
            self.right_hand.step(control_cmd['right_hand'])
        if self.left_leg and 'left_leg' in control_cmd:
            self.left_leg.step(control_cmd['left_leg'])
        if self.right_leg and 'right_leg' in control_cmd:
            self.right_leg.step(control_cmd['right_leg'])
        if self.torso and 'torso' in control_cmd:
            self.torso.step(control_cmd['torso'])

    @property
    def state(self):
        """
        Get the current state of the robot.
        :return: A dictionary containing the state of each body part.
        """
        state = {}
        if self.head:
            state['head'] = self.head.state
        if self.left_arm:
            state['left_arm'] = self.left_arm.state
        if self.right_arm:
            state['right_arm'] = self.right_arm.state
        if self.left_hand:
            state['left_hand'] = self.left_hand.state
        if self.right_hand:
            state['right_hand'] = self.right_hand.state
        if self.left_leg:
            state['left_leg'] = self.left_leg.state
        if self.right_leg:
            state['right_leg'] = self.right_leg.state
        if self.torso:
            state['torso'] = self.torso.state
        return state
