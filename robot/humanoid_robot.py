from omegaconf import OmegaConf
from typing import Dict
from .robot_components import RobotComponents
import numpy as np
import time

class HumanoidRobot:

    def __init__(self, config_path):
        config = OmegaConf.load(config_path)
        self.head = RobotComponents[config.head.type](name='head')
        self.left_arm = RobotComponents[config.left_arm.type](name='left_arm', config=config.config)
        self.right_arm = RobotComponents[config.right_arm.type](name='right_arm', config=config.config)
        self.left_hand = RobotComponents[config.left_hand.type](name='left_hand')
        self.right_hand = RobotComponents[config.right_hand.type](name='right_hand')
        self.left_leg = RobotComponents[config.left_leg.type](name='left_leg')
        self.right_leg = RobotComponents[config.right_leg.type](name='right_leg')
        self.torso = RobotComponents[config.torso.type](name='torso')
        self.components = {c.name: c for c in [self.head, self.left_arm, self.right_arm, self.left_hand, self.right_hand, self.left_leg, self.right_leg, self.torso] if c is not None}

        # Should be set at the first time received cmd
        self.left_arm_init_pos = None
        self.right_arm_init_pos = None

    def step(self, control_cmd: Dict):
        """
        Update the robot's state based on control commands.
        :param control_cmd: A dictionary containing control commands for different body parts.
        """
        for name, cmd in control_cmd.items():
            if name == 'left_arm':
                cmd = np.array(cmd, dtype=float)
                if self.left_arm_init_pos is None:
                    self.left_arm_init_pos = cmd[:3].copy()
                    control_cmd[name] = None
                else:
                    cmd[:3] -= self.left_arm_init_pos
                    control_cmd[name] = cmd
            if name == 'right_arm':
                cmd = np.array(cmd, dtype=float)
                if self.right_arm_init_pos is None:
                    self.right_arm_init_pos = cmd[:3].copy()
                    control_cmd[name] = None
                else:
                    cmd[:3] -= self.right_arm_init_pos
                    control_cmd[name] = cmd
        for name, c in self.components.items():
            # print(f'input {name} cmd: {control_cmd.get(name, None)}')
            c.step(control_cmd.get(name, None))
        time.sleep(0.01)

    def reset(self):
        for name, c in self.components.items():
            c.reset()

    def go_default(self):
        for name, c in self.components.items():
            c.go_default()
        print('Waiting robot go default ...')
        time.sleep(0.1)
        while not all([c.done for c in self.components.values()]):
            time.sleep(0.1)


    def go_zero(self):
        for name, c in self.components.items():
            c.go_zero()
        print('Waiting robot go zero ...')
        time.sleep(0.1)
        while not all([c.done for c in self.components.values()]):
            time.sleep(0.1)

    @property
    def state(self):
        """
        Get the current state of the robot.
        :return: A dictionary containing the state of each body part.
        """
        state = {}
        for name, c in self.components.items():
            state[name] = c.state
        return state
