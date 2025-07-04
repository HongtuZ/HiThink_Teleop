from pathlib import Path
from avp_stream import VisionProStreamer as AVPStreamer

from robot.humanoid_robot import HumanoidRobot
import teleop.helper as helper

def avp_data2hithink_robot_cmd(avp_data):
    robot_cmd = {}
    # Head
    # Left Arm
    left_arm_cmd, right_arm_cmd = helper.avp2agilex(avp_data)
    robot_cmd['left_arm'], robot_cmd['right_arm'] = left_arm_cmd, right_arm_cmd
    # Hands
    # Torso
    # Legs
    # print(robot_cmd)
    return robot_cmd

def avp_teleop():
    streamer = AVPStreamer(ip='192.168.31.90', record=False)
    hithink_robot = HumanoidRobot(config_path=str(Path(__file__).parent/'config/hithink_robot.yaml'))
    while True:
        avp_data = streamer.get_latest()
        hithink_robot.step(avp_data2hithink_robot_cmd(avp_data))

if __name__ == '__main__':
    avp_teleop()