from threading import Lock
from typing import List, Optional, Dict

import numpy as np
import pinocchio as pin
import yaml

class PinocchioMotionControl():
    def __init__(
        self,
        urdf_path: str,
        ee_name: str,
        ik_dt: float,
        ik_damping: float,
        ik_eps: float,
    ):
        self._qpos_lock = Lock()

        # Config
        self.ee_name = ee_name
        self.dt = ik_dt
        self.ik_damping = ik_damping * np.eye(6)
        self.ik_eps = ik_eps

        # Build robot
        self.model: pin.Model = pin.buildModelFromUrdf(urdf_path)
        self.data: pin.Data = self.model.createData()
        frame_mapping: Dict[str, int] = {}

        for i, frame in enumerate(self.model.frames):
            frame_mapping[frame.name] = i

        if self.ee_name not in frame_mapping:
            raise ValueError(
                f"End effector name {self.ee_name} not find in robot with path: {urdf_path}."
            )
        self.frame_mapping = frame_mapping
        self.ee_frame_id = frame_mapping[self.ee_name]

        self.lower_limit = self.model.lowerPositionLimit
        self.upper_limit = self.model.upperPositionLimit

        # Current state
        self.qpos = pin.neutral(self.model)
        pin.forwardKinematics(self.model, self.data, self.qpos)
        self.ee_pose: pin.SE3 = pin.updateFramePlacement(
            self.model, self.data, self.ee_frame_id
        )

    def step(self, pose: Optional[np.ndarray], repeat=1):
        oMdes = pin.XYZQUATToSE3(pose)
        with self._qpos_lock:
            qpos = self.qpos.copy()

        success = False
        for k in range(100 * repeat):
            pin.forwardKinematics(self.model, self.data, qpos)
            ee_pose = pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
            iMd = ee_pose.actInv(oMdes)
            err = pin.log(iMd).vector
            if np.linalg.norm(err) < self.ik_eps:
                success = True
                break

            J = pin.computeFrameJacobian(self.model, self.data, qpos, self.ee_frame_id)
            J = -np.dot(pin.Jlog6(iMd.inverse()), J)

            v = -J.T.dot(np.linalg.solve(J.dot(J.T) + self.ik_damping*np.eye(6), err))
            qpos = pin.integrate(self.model, qpos, v * self.dt)

        success = np.all(qpos >= self.lower_limit) and np.all(qpos <= self.upper_limit)
        qpos = np.clip(qpos, self.lower_limit, self.upper_limit)
        print('-----------------')
        print(self.ee_name, ' IK:', success, ' err:', np.linalg.norm(err))
        # self.set_current_qpos(qpos) # done outside
        return success, qpos

    def compute_ee_pose(self, qpos: np.ndarray) -> np.ndarray:
        pin.forwardKinematics(self.model, self.data, qpos)
        oMf: pin.SE3 = pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
        xyzw_pose = pin.SE3ToXYZQUAT(oMf)
        return xyzw_pose

    def get_current_qpos(self) -> np.ndarray:
        with self._qpos_lock:
            return self.qpos.copy()

    def set_current_qpos(self, qpos: np.ndarray):
        with self._qpos_lock:
            self.qpos = qpos
            pin.forwardKinematics(self.model, self.data, self.qpos)
            self.ee_pose = pin.updateFramePlacement(
                self.model, self.data, self.ee_frame_id
            )

    def get_ee_name(self) -> str:
        return self.ee_name

    def get_dof(self) -> int:
        return pin.neutral(self.model).shape[0]

    def get_timestep(self) -> float:
        return self.dt

    def get_joint_names(self) -> List[str]:
        # Pinocchio by default add a dummy joint name called "universe"
        names = list(self.model.names)
        return names[1:]

    def is_use_gpu(self) -> bool:
        return False
