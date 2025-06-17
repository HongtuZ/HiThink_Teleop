# Teleop by visionPro
import yaml
from pathlib import Path
from avp_stream import VisionProStreamer as AVPStreamer
from robot.humanoid_robot import HumanoidRobot
import numpy as np

def avp_data2target_pose(avp_data):
    target_pose = TargetPose()
    target_pose.left_wrist = avp_data['left_wrist']
    target_pose.right_wrist = avp_data['right_wrist']
    target_pose.left_fingers = avp_data['left_fingers']
    target_pose.right_fingers = avp_data['right_fingers']
    return target_pose

def avp_teleop():
    streamer = AVPStreamer(ip='192.168.31.22', record=False)
    robot = HumanoidRobot(config_path=str(Path(__file__).parent/'config/hithink_robot.yaml'))
    while True:
        data = streamer.get_latest()
        robot.step(avp_data2target_pose(data))

def test():
    pickle_data = np.load('../assets/offline_avp_stream.pkl', allow_pickle=True)
    for data in pickle_data:
        print(data.keys())
        break

if __name__ == '__main__':
    # avp_teleop()
    test()