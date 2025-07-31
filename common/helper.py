import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

def avp2agilex(avp_data):
    # Left hand
    lw2ee_rot_transform = np.array([[1,0,0],
                                    [0,-1,0],
                                    [0,0,-1]])
    l_ee_transform = np.array([[0, 1, 0, 0],
                               [-1, 0, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]], dtype=np.float64)
    l_ee_pose = l_ee_transform @ np.array(avp_data['left_wrist'][0])
    l_ee_pose[:3, :3] = l_ee_pose[:3, :3] @ lw2ee_rot_transform
    l_ee_pos_xyz = l_ee_pose[:3, 3]
    l_ee_quat_xyzw = mat2quat(l_ee_pose[:3, :3])
    l_gripper_range = (np.clip(avp_data['left_pinch_distance'], 0.02, 0.12) - 0.02)/0.1 
    l_ee_cmd =  np.concatenate([l_ee_pos_xyz, l_ee_quat_xyzw, [l_gripper_range]], axis=0)

    
    # print('left_ee:', [f'{v:.2f}' for v in left_arm_cmd])
    # Right Arm
    rw2ee_rot_transform = np.array([[-1,0,0],
                                    [0,-1,0],
                                    [0,0,1]])
    r_ee_transform = np.array([[0, 1, 0, 0],
                               [-1, 0, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]], dtype=np.float64)
    r_ee_pose = r_ee_transform @ np.array(avp_data['right_wrist'][0])
    r_ee_pose[:3,:3] = r_ee_pose[:3,:3] @ rw2ee_rot_transform
    r_ee_pos_xyz = r_ee_pose[:3, 3]
    r_ee_quat_xyzw = mat2quat(r_ee_pose[:3, :3])
    r_gripper_range = (np.clip(avp_data['right_pinch_distance'], 0.02, 0.12) - 0.02)/0.1
    r_ee_cmd =  np.concatenate([r_ee_pos_xyz, r_ee_quat_xyzw, [r_gripper_range]], axis=0)

    return l_ee_cmd, r_ee_cmd

def smooth_pose(old_pose,  new_pose, alpha=0.5):
    """
    input:
        old_pose: (x, y, z, quat_x, quat_y, quat_z, quat_w)
        new_pose: (x, y, z, quat_x, quat_y, quat_z, quat_w)
        alpha: 平滑系数(0~1)，越小越平滑，响应越慢
    return:
        smoothed_pose: (x, y, z, quat_x, quat_y, quat_z, quat_w)
    """
    smoothed_pos = (1 - alpha) * old_pose[:3] + alpha * new_pose[:3]

    # --- 平滑姿态（四元数） ---
    # 注意：scipy 的四元数顺序是 [x, y, z, w]
    slerp = Slerp([0, 1], R.concatenate([R.from_quat(old_pose[-4:]), R.from_quat(new_pose[-4:])]))
    interp_r = slerp([alpha])[0]  # 插值到 alpha 点
    smoothed_quat = interp_r.as_quat()  # [x, y, z, w]

    return np.concatenate([smoothed_pos, smoothed_quat], axis=0)

def mat2quat(mat):
    # return quat: [x, y, z, w]
    r = R.from_matrix(mat[:3,:3])
    q = r.as_quat()  # [x, y, z, w]
    return q