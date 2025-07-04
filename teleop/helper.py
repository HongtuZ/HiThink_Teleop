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
    l_gripper_range = (np.clip(avp_data['left_pinch_distance'], 0.02, 0.12) - 0.02)/0.1 
    lrx, lry, lrz = t3d.euler.mat2euler(l_ee_pose)
    left_arm_cmd = [l_ee_pose[0, 3], l_ee_pose[1, 3], l_ee_pose[2, 3],
                             np.rad2deg(lrx), np.rad2deg(lry), np.rad2deg(lrz), l_gripper_range]
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
    r_gripper_range = (np.clip(avp_data['right_pinch_distance'], 0.02, 0.12) - 0.02)/0.1
    rrx, rry, rrz = t3d.euler.mat2euler(r_ee_pose)
    right_arm_cmd = [r_ee_pose[0, 3], r_ee_pose[1, 3], r_ee_pose[2, 3],
                              np.rad2deg(rrx), np.rad2deg(rry), np.rad2deg(rrz), r_gripper_range]
    # print('right_ee:', [f'{v:.2f}' for v in right_arm_cmd])
    return np.array(left_arm_cmd), np.array(right_arm_cmd)