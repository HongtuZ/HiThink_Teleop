from robot.humanoid_robot import HumanoidRobot
from pprint import pprint
import time

if __name__ == "__main__":
    robot = HumanoidRobot(config_path='./config/hithink_robot.yaml')
    while True:
        print(robot.state)
        time.sleep(1)
        # robot.go_default()
        # robot.go_zero()
        # robot.go_default()
    # start_time = time.perf_counter()
    # start_ee_pose = robot.right_arm.ee_pose
    # while True:
    #     duration_time = time.perf_counter() - start_time
    #     z = duration_time * 0.0001
    #     target_ee_pose = start_ee_pose
    #     target_ee_pose[2] += z
    #     robot.right_arm.step(target_ee_pose)