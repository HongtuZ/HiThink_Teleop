import yaml
from typing import Dict
from .robot_components import RobotComponents

class HumanoidRobot:

    def __init__(self, config_path):
        config = yaml.safe_load(open(config_path), 'r')
        self.head = RobotComponents[config.head.type]() if config.head else None
        self.left_arm = RobotComponents[config.left_arm.type](can_id=config.left_arm.can_id, is_left=True) if config.left_arm else None
        self.right_arm = RobotComponents[config.right_arm.type](can_id=config.right_arm.can_id, is_left=False) if config.right_arm else None
        self.left_hand = RobotComponents[config.left_hand.type]() if config.left_hand else None
        self.right_hand = RobotComponents[config.right_hand.type]() if config.right_hand else None
        self.left_leg = RobotComponents[config.left_leg.type]() if config.left_leg else None
        self.right_leg = RobotComponents[config.right_leg.type]() if config.right_leg else None
        self.torso = RobotComponents[config.torso.type]() if config.torso else None

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