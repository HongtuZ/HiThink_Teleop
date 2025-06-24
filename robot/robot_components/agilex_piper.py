import time
import numpy as np
from piper_sdk import *

ArmState = {
            0x00: '正常',
            0x01: '急停',
            0x02: '无解',
            0x03: '奇异点',
            0x04: '目标角度超过限',
            0x05: '关节通信异常',
            0x06: '关节抱闸未打开',
            0x07: '机械臂发生碰撞',
            0x08: '拖动示教时超速',
            0x09: '关节状态异常',
            0x0A: '其它异常',
            0x0B: '示教记录',
            0x0C: '示教执行',
            0x0D: '示教暂停',
            0x0E: '主控NTC过温',
            0x0F: '释放电阻NTC过温',
        }

class AgilexPiper:
    
    def __init__(self, can_id, is_left=False):
        self.can_id = can_id
        self.piper = C_PiperInterface_V2(can_id)
        self.enabled = self.init()

    def init(self):
        self.piper.ConnectPort()
        is_enabled, is_timeout = False, False
        start_time = time.perf_counter()
        while not (is_enabled or is_timeout):
            is_enabled = self.piper.EnablePiper()
            is_timeout = time.perf_counter() - start_time > 5 # 5 seconds timeout
            time.sleep(0.01)
        return is_enabled

    def step(self, ee_pose, gripper_range):
        # EE Pose: [X, Y, Z, RX, RY, RZ, gripper_range] m, degree and (0,1)
        # gripper_range: [0, 1] (0: open, 1: close)
        if not self.enabled:
            return
        print(f"{self.can_id} cmd: {cmd}")
        gripper_range = (np.clip(cmd[-1], 0, 1)*1e5).astype(int)
        ee_pose = (np.array(cmd[:6], dtype=float)*1e6).astype(int)
        x, y, z, rx, ry, rz = ee_pose.tolist()
        self.piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        self.piper.EndPoseCtrl(round(x), round(y), round(z), round(rx), round(ry), round(rz))
        self.piper.GripperCtrl(gripper_range, 1000, 0x03, 0)
        print(f'{self.can_id} res: {ArmState[self.piper.GetArmStatus().arm_status.arm_status]}')
        time.sleep(0.01)

    def go_default(self):
        if not self.enabled:
            return
        self.piper.GripperCtrl(0,1000,0x01, 0)
        factor = 57295.7795 #1000*180/3.1415926
        position = [1.57, 3.0, -3.0 ,0, 0, 0, 0.08]
        joint_0 = round(position[0]*factor)
        joint_1 = round(position[1]*factor)
        joint_2 = round(position[2]*factor)
        joint_3 = round(position[3]*factor)
        joint_4 = round(position[4]*factor)
        joint_5 = round(position[5]*factor)
        joint_6 = round(position[6]*1000*1000)
        self.piper.MotionCtrl_2(0x01, 0x01, 30, 0x00)
        self.piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
        self.piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
        time.sleep(0.2)
        start_time = time.perf_counter()
        while True:
            if self.piper.GetArmStatus().arm_status.motion_status == 0x00 or time.perf_counter() - start_time > 10:
                break
            time.sleep(0.1)


    def go_zero(self):
        if not self.enabled:
            return
        self.piper.MotionCtrl_2(0x01, 0x01, 30, 0x00)
        self.piper.JointCtrl(0, 0, 0, 0, 0, 0)
        self.piper.GripperCtrl(0, 1000, 0x01, 0)
        time.sleep(0.2)
        start_time = time.perf_counter()
        while True:
            if self.piper.GetArmStatus().arm_status.motion_status == 0x00 or time.perf_counter() - start_time > 10:
                break
            time.sleep(0.1)
            

    def reset(self):
        # Reset the Piper
        self.piper.MotionCtrl_1(0x02,0,0)#恢复
        self.piper.MotionCtrl_2(0, 0, 0, 0x00)#位置速度模式

    @property
    def state(self):
        if not self.ready:
            return "Not ready"
        return {
            # 'Status': str(self.piper.GetArmStatus()),
            'Joints': str(self.piper.GetArmJointMsgs()),
            'Gripper': str(self.piper.GetArmGripperMsgs()),
            'EndPose': str(self.piper.GetArmEndPoseMsgs()),
        }
    
    @property
    def ee_pose(self):
        griper_range = self.piper.GetArmGripperMsgs().gripper_state.grippers_angle * 10
        ee_pose = self.piper.GetArmEndPoseMsgs().end_pose
        return np.array([ee_pose.X_axis, ee_pose.Y_axis, ee_pose.Z_axis, ee_pose.RX_axis, ee_pose.RY_axis, ee_pose.RZ_axis, griper_range]) * 1e-6