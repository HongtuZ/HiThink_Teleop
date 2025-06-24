import numpy as np
import transforms3d as t3d

from pathlib import Path
from avp_stream import VisionProStreamer as AVPStreamer

from robot.humanoid_robot import HumanoidRobot

def avp_data2hithink_robot_cmd(avp_data):
    robot_cmd = {}
    # Head
    # Left Arm
    l_ee_transform = np.array([[0, 1, 0, 0],
                               [0, 0, -1, -0.2],
                               [-1, 0, 0, -0.04],
                               [0, 0, 0, 1]], dtype=np.float64)
    l_ee_pose = l_ee_transform @ avp_data['left_wrist'][0]
    l_gripper_range = (np.clip(avp_data['left_pinch_distance'], 0.02, 0.12) - 0.02)/0.1 
    lrx, lry, lrz = t3d.euler.mat2euler(l_ee_pose)
    robot_cmd['left_arm'] = [l_ee_pose[0, 3], l_ee_pose[1, 3], l_ee_pose[2, 3],
                             np.rad2deg(lrx), np.rad2deg(lry), np.rad2deg(lrz), l_gripper_range]
    # Right Arm
    r_ee_transform = np.array([[0, 1, 0, 0],
                               [0, 0, 1, 0.2],
                               [1, 0, 0, -0.04],
                               [0, 0, 0, 1]], dtype=np.float64)
    r_ee_pose = r_ee_transform @ avp_data['right_wrist'][0]
    r_gripper_range = (np.clip(avp_data['right_pinch_distance'], 0.02, 0.12) - 0.02)/0.1
    rrx, rry, rrz = t3d.euler.mat2euler(r_ee_pose)
    robot_cmd['right_arm'] = [r_ee_pose[0, 3], r_ee_pose[1, 3], r_ee_pose[2, 3],
                              np.rad2deg(rrx), np.rad2deg(rry), np.rad2deg(rrz), r_gripper_range]
    # Hands
    # Torso
    # Legs
    return robot_cmd

def avp_teleop():
    streamer = AVPStreamer(ip='192.168.31.22', record=False)
    hithink_robot = HumanoidRobot(config_path=str(Path(__file__).parent/'config/hithink_robot.yaml'))
    while True:
        avp_data = streamer.get_latest()
        hithink_robot.step(avp_data2hithink_robot_cmd(avp_data))

if __name__ == '__main__':
    avp_teleop()