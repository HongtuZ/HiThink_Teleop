import time
import numpy as np
from piper_sdk import *

class AgilexPiper:
    
    def __init__(self, can_id, is_left=False):
        self.can_id = can_id
        self.is_left = is_left
        self.piper = C_PiperInterface_V2(can_id)
        self.init()
        if self.is_left:
            self.piper.MotionCtrl_2(0x00, 0x00, 100, 0x02)
            self.piper.MotionCtrl_2(0x01, 0x00, 100, 0x02)
        else:
            self.piper.MotionCtrl_2(0x00, 0x00, 100, 0x03)
            self.piper.MotionCtrl_2(0x01, 0x00, 100, 0x03)

    def init(self):
        timeout = 5
        is_enabled, is_timeout = False, False
        start_time = time.time()
        while not (is_enabled or is_timeout):
            time.sleep(1)
            is_enabled = self.piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status and \
                self.piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status and \
                self.piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status and \
                self.piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status and \
                self.piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status and \
                self.piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
            print(f"{self.can_id} 使能状态: {is_enabled}")
            self.piper.EnableArm(7)
            # 检查是否超过超时时间
            if time.time() - start_time > timeout:
                print(f"{self.can_id} 超时....")
                is_timeout = True
        if not is_enabled and is_timeout:
            print(f"{self.can_id} 程序自动使能超时,退出程序")
            exit(0)

    def step(self, cmd):
        # EE Pose: [X, Y, Z, RX, RY, RZ, gripper_range] m, degree and (0,1)
        # gripper_range: [0, 1] (0: open, 1: close)
        gripper_range = (np.clip(cmd[-1], 0, 1)*100*1000).astype(int)
        ee_pose = (np.array(ee_pose[:6], dtype=float)*1000*1000).astype(int)
        x, y, z, rx, ry, rz = ee_pose.tolist()
        self.piper.EndPoseCtrl(x, y, z, rx, ry, rz)
        self.piper.GripperCtrl(gripper_range, 1000, 0x01, 0)

    def state(self):
        return {
            'Status': self.piper.GetArmStatus(),
            'Joints': self.piper.GetArmJointMsgs(),
            'Gripper': self.piper.GetArmGripperMsgs(),
        }