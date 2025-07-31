import common.helper as helper

from pathlib import Path
from avp_stream import VisionProStreamer as AVPStreamer
from teleop_sim import Sim
from omegaconf import OmegaConf

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

def avp_teleop(config_path:str):
    config = OmegaConf.load(config_path)
    streamer = AVPStreamer(ip=config.avp.ip, record=False)
    sim = Sim(config.left_arm.config)
    while True:
        avp_data = streamer.get_latest()
        sim.step(avp_data2hithink_robot_cmd(avp_data))

if __name__ == '__main__':
    config_path = Path(__file__).parent/'config/hithink_robot.yaml'
    avp_teleop(str(config_path))