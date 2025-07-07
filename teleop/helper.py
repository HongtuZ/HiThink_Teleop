import numpy as np
import transforms3d as t3d

def avp2agilex(avp_data):
    # Left hand
    lw2ee_rot_transform = np.array([[0,0,1],
                                    [0,-1,0],
                                    [1,0,0]])
    l_ee_transform = np.array([[0, 1, 0, 0],
                               [-1, 0, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]], dtype=np.float64)
    l_ee_pose = l_ee_transform @ np.array(avp_data['left_wrist'][0])
    l_ee_pose[:3, :3] = l_ee_pose[:3, :3] @ lw2ee_rot_transform
    l_ee_pos_xyz = l_ee_pose[:3, 3]
    l_ee_quat_wxyz = t3d.quaternions.mat2quat(l_ee_pose[:3, :3])
    l_gripper_range = (np.clip(avp_data['left_pinch_distance'], 0.02, 0.12) - 0.02)/0.1 
    l_ee_cmd =  np.concatenate([l_ee_pos_xyz, l_ee_quat_wxyz, [l_gripper_range]], axis=0)

    
    # print('left_ee:', [f'{v:.2f}' for v in left_arm_cmd])
    # Right Arm
    rw2ee_rot_transform = np.array([[0,0,-1],
                                    [0,-1,0],
                                    [-1,0,0]])
    r_ee_transform = np.array([[0, 1, 0, 0],
                               [-1, 0, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]], dtype=np.float64)
    r_ee_pose = r_ee_transform @ np.array(avp_data['right_wrist'][0])
    r_ee_pose[:3,:3] = r_ee_pose[:3,:3] @ rw2ee_rot_transform
    r_ee_pos_xyz = r_ee_pose[:3, 3]
    r_ee_quat_wxyz = t3d.quaternions.mat2quat(r_ee_pose[:3, :3])
    r_gripper_range = (np.clip(avp_data['right_pinch_distance'], 0.02, 0.12) - 0.02)/0.1
    r_ee_cmd =  np.concatenate([r_ee_pos_xyz, r_ee_quat_wxyz, [r_gripper_range]], axis=0)

    # print('right_ee:', [f'{v:.2f}' for v in right_arm_cmd])
    return l_ee_cmd, r_ee_cmd