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

CtrlMode = {
            0x00: '待机模式',
            0x01: 'CAN指令控制模式',
            0x02: '示教模式',
            0x03: '以太网控制模式',
            0x04: 'wifi控制模式',
            0x05: '遥控器控制模式',
            0x06: '联动示教输入模式',
            0x07: '离线轨迹模式',
}

MotionMode = {
            0x00: '到达指定点位',
            0x01: '未到达指定点位',
}

TeachMode = {
            0x00: '关闭',
            0x01: '开始示教记录（进入拖动示教模式）',
            0x02: '结束示教记录（退出拖动示教模式）',
            0x03: '执行示教轨迹（拖动示教轨迹复现）',
            0x04: '暂停执行',
            0x05: '继续执行（轨迹复现继续）',
            0x06: '终止执行',
            0x07: '运动到轨迹起点',
}

ModeFbk = {
            0x00: 'MOVE P',
            0x01: 'MOVE J',
            0x02: 'MOVE L',
            0x03: 'MOVE C',
            0x04: 'MOVE M',
}

class AgilexPiper:
    
    def __init__(self, name, can_id, is_left=False):
        self.name = name
        self.can_id = can_id
        self.piper = C_PiperInterface_V2(can_id)
        self.init()
        self.default_pose = [0, 0, 0 ,0, 0, 0, 0.08] if is_left else [0, 0, 0 ,0, 0, 0, 0.08]
        self.init_ee_pose = self.ee_pose

    def init(self):
        self.piper.ConnectPort()
        is_enabled, is_timeout = False, False
        start_time = time.perf_counter()
        while not (is_enabled or is_timeout):
            is_enabled = self.piper.EnablePiper()
            is_timeout = time.perf_counter() - start_time > 5 # 5 seconds timeout
            time.sleep(0.01)
        if not is_enabled:
            print(f'Error: {self.name} on {self.can_id} is not enabled!')
            exit(1)

    def step(self, cmd):
        # cmd: [X, Y, Z, RX, RY, RZ, gripper]: Delta x y z
        # unit: X,Y,Z: m, RX,RY,RZ: degree, gripper: (0, 1)
        if cmd is None:
            return
        print('----------------------------------------------------')
        print(f'cur ee pose:', [f'{v:.2f}' for v in self.ee_pose])
        print(f"input {self.name} cmd:", [f'{v:.2f}' for v in cmd])
        cmd[:3] += self.init_ee_pose[:3]
        print(f"after {self.name} cmd:", [f'{v:.2f}' for v in cmd])
        gripper_range = (np.clip(cmd[-1], 0, 1)*1e5).astype(int)
        x, y, z = cmd[:3]*1e6 # m -> 0.001mm
        rx, ry, rz = cmd[3:6]*1e3 # degree -> 0.001degree
        self.piper.ModeCtrl(0x01, 0x00, 100, 0x00)
        self.piper.EndPoseCtrl(round(x), round(y), round(z), round(rx), round(ry), round(rz))
        self.piper.GripperCtrl(gripper_range, 1000, 0x03, 0)
        print(f'{self.name} res: {ArmState[self.piper.GetArmStatus().arm_status.arm_status]}')

    def go_default(self):
        print(f'{self.name}: Go default')
        factor = 57295.7795 #1000*180/3.1415926
        joints = list(map(lambda x: round(x*factor), self.default_pose[:6]))
        gripper_range = round(self.default_pose[-1]*1e6)
        self.piper.ModeCtrl(0x01, 0x01, 30, 0x00)
        self.piper.JointCtrl(*joints)
        self.piper.GripperCtrl(abs(gripper_range), 1000, 0x01, 0)


    def go_zero(self):
        self.piper.ModeCtrl(0x01, 0x01, 30, 0x00)
        self.piper.JointCtrl(0, 0, 0, 0, 0, 0)
        self.piper.GripperCtrl(0, 1000, 0x01, 0)
            

    def reset(self):
        # Reset the Piper
        print(f'{self.name}: Stop')
        self.piper.MotionCtrl_1(0x01,0,0)
        time.sleep(3)
        print(f'{self.name}: Recover')
        self.piper.MotionCtrl_1(0x02,0,0)#恢复
        while True:
            self.piper.EnablePiper()
            self.piper.ModeCtrl(1, 1, 100, 0)
            piper_state = self.piper.GetArmStatus().arm_status
            if piper_state.ctrl_mode == 1:
                break
            print(f'Try to change the control mode from {piper_state.ctrl_mode} to {1} ...')
            time.sleep(0.1)

    @property
    def state(self):
        arm_status = self.piper.GetArmStatus().arm_status
        status = f'机械臂状态: {ArmState[arm_status.arm_status]} 控制模式: {CtrlMode[arm_status.ctrl_mode]} {ModeFbk[arm_status.mode_feed]} 运动状态: {MotionMode[arm_status.motion_status]} 示教状态: {TeachMode[arm_status.teach_status]}' 
        return {
            'Status': status,
            # 'Joints': str(self.piper.GetArmJointMsgs()),
            # 'Gripper': str(self.piper.GetArmGripperMsgs()),
            'EndPose': self.ee_pose,
        }

    @property
    def done(self):
        return self.piper.GetArmStatus().arm_status.motion_status == 0x00
    
    @property
    def ee_pose(self):
        griper_range = self.piper.GetArmGripperMsgs().gripper_state.grippers_angle * 1e-5 # -> (0,1)
        ee_pose = self.piper.GetArmEndPoseMsgs().end_pose
        ee_xyz = np.array([ee_pose.X_axis, ee_pose.Y_axis, ee_pose.Z_axis], dtype=float)*1e-6 # 0.001mm -> m
        ee_rxyz = np.array([ee_pose.RX_axis, ee_pose.RY_axis, ee_pose.RZ_axis], dtype=float)*1e-3 # 0.001degree -> degree
        return np.concatenate([ee_xyz, ee_rxyz, [griper_range]], axis=0)