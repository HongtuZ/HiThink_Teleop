from robot.humanoid_robot import HumanoidRobot
from pprint import pprint
import time

if __name__ == "__main__":
    robot = HumanoidRobot(config_path='./config/hithink_robot.yaml')
    robot.reset()