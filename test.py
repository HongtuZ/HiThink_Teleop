from robot.humanoid_robot import HumanoidRobot
import time
import numpy as np
from pprint import pprint
from pathlib import Path
from omegaconf import OmegaConf

if __name__ == "__main__":
    config = OmegaConf.load(Path(__file__).parent/'config/hithink_robot.yaml')
    hithink_robot = HumanoidRobot(config)
    hithink_robot.go_default()