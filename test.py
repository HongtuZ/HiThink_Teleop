# from robot.humanoid_robot import HumanoidRobot
import time
import numpy as np
from pprint import pprint
from robot.robot_components.motion_controller.pinocchio_motion_control import PinocchioMotionControl

if __name__ == "__main__":
    # robot = HumanoidRobot(config_path='./config/hithink_robot.yaml')
    # while True:
    #     print(robot.state)
    #     time.sleep(1)
    pin_controller = PinocchioMotionControl(
        urdf_path='/home/robot/hongtu/HiThink_Teleop/assets/urdf/piper_arm/piper_description.urdf',
        ee_name='gripper_base',
        ik_dt=0.1,
        ik_damping=0.01,
        ik_eps=1e-3,
    )
    pin_controller.step(pos=np.array([0.3, 0.0, 0.3]), quat=np.array([1, 0, 0, 0]), repeat=1)