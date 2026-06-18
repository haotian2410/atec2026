"""B2Piper / Piper 机械臂运动学先验。

常量来自 demo/solution_E.py 中已经验证过的 Piper USD 派生实现。本模块不依赖
Isaac 运行时，可用于末端相机点投影、抓取预成形，以及后续基于 IK 的机械臂控制。
"""

from __future__ import annotations

import numpy as np


ARM_JOINT_NAMES = [
    "arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5", "arm_joint6",
]
GRIPPER_JOINT_NAMES = ["arm_joint7", "arm_joint8"]
ALL_ARM_JOINT_NAMES = ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES

DEFAULT_ARM_QPOS = np.array([0.0, 1.2, -1.5, 0.0, 1.2, 0.0], dtype=np.float64)
DEFAULT_GRIPPER_QPOS = np.array([0.035, -0.035], dtype=np.float64)
ACTION_SCALE = 0.5

Q_LOWER = np.deg2rad([-150.0, 0.0, -170.0, -100.0, -69.9, -120.0])
Q_UPPER = np.deg2rad([124.2, 179.9, 0.0, 100.0, 69.9, 120.0])
EXPECTED_ZERO_FK_POS = np.array([0.0561424668, -0.0000000566, 0.213193122], dtype=np.float64)


def quat_to_mat(q):
    # USD/Isaac 中这里使用标量在前的四元数 wxyz。
    w, x, y, z = np.asarray(q, dtype=np.float64)
    n = max(np.sqrt(w * w + x * x + y * y + z * z), 1e-12)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def transform(pos, quat):
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = quat_to_mat(quat)
    mat[:3, 3] = np.asarray(pos, dtype=np.float64)
    return mat


def rotz(angle):
    c, s = np.cos(angle), np.sin(angle)
    mat = np.eye(4, dtype=np.float64)
    mat[0, 0], mat[0, 1] = c, -s
    mat[1, 0], mat[1, 1] = s, c
    return mat


def rot_error(r_cur, r_des):
    re = r_des @ r_cur.T
    return 0.5 * np.array([
        re[2, 1] - re[1, 2],
        re[0, 2] - re[2, 0],
        re[1, 0] - re[0, 1],
    ], dtype=np.float64)


class PiperKinematics:
    """USD 派生的 base_link -> gripper_base FK/IK，结果位于机械臂基座坐标系。"""

    JOINTS = [
        ((0.0, 0.0, 0.123), (1.0, 0.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (0.04800847, -0.048013408, -0.7054761, -0.7054739)),
        ((0.28503, 0.0, 0.0), (0.6239962, 0.0, 0.0, -0.7814274)),
        ((-0.021984, -0.25075, 0.0), (0.7071055, 0.7071081, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (0.7071055, -0.7071081, 0.0, 0.0)),
        ((0.000088259, -0.091, 0.0), (0.7071055, 0.7071081, 0.0, 0.0)),
    ]
    FIXED = [transform(p, q) for p, q in JOINTS]

    def fk(self, q):
        # 逐关节累乘固定变换和绕 z 轴的关节转角，得到末端齐次变换。
        q = np.asarray(q, dtype=np.float64)
        mat = np.eye(4, dtype=np.float64)
        for i in range(6):
            mat = mat @ self.FIXED[i] @ rotz(q[i])
        return mat

    def fk_pos(self, q):
        return self.fk(q)[:3, 3]

    def jacobian(self, q, eps=1e-6):
        # 数值差分 Jacobian：足够用于策略层粗 IK，避免引入额外机器人库。
        q = np.asarray(q, dtype=np.float64)
        m0 = self.fk(q)
        p0, r0 = m0[:3, 3], m0[:3, :3]
        jac = np.zeros((6, 6), dtype=np.float64)
        for j in range(6):
            dq = q.copy()
            dq[j] += eps
            mj = self.fk(dq)
            jac[:3, j] = (mj[:3, 3] - p0) / eps
            jac[3:, j] = rot_error(r0, mj[:3, :3]) / eps
        return jac

    def ik(self, target_pos, target_r=None, q_init=None, iters=250, lam=0.05, ko=0.5, polish=40, step_clip=0.2):
        # 阻尼最小二乘 IK：先主要收敛位置，若给定 target_r 再用零空间补姿态。
        q = np.asarray(q_init, dtype=np.float64).copy() if q_init is not None else np.zeros(6, dtype=np.float64)
        target_pos = np.asarray(target_pos, dtype=np.float64)
        use_ori = target_r is not None
        eye = np.eye(6, dtype=np.float64)
        for _ in range(iters):
            mat = self.fk(q)
            ep = target_pos - mat[:3, 3]
            jac = self.jacobian(q)
            jp = jac[:3]
            jjt = jp @ jp.T
            jp_sharp = jp.T @ np.linalg.inv(jjt + (lam * lam) * np.eye(3))
            dq = jp_sharp @ ep
            if use_ori:
                null = eye - jp_sharp @ jp
                eo = rot_error(mat[:3, :3], target_r)
                dq = dq + null @ (ko * (jac[3:].T @ eo))
            q = np.clip(q + np.clip(dq, -step_clip, step_clip), Q_LOWER, Q_UPPER)
        for _ in range(polish):
            mat = self.fk(q)
            ep = target_pos - mat[:3, 3]
            if np.linalg.norm(ep) < 1e-7:
                break
            jp = self.jacobian(q)[:3]
            jjt = jp @ jp.T
            dq = jp.T @ np.linalg.solve(jjt + (lam * lam) * np.eye(3), ep)
            q = np.clip(q + np.clip(dq, -step_clip, step_clip), Q_LOWER, Q_UPPER)
        return np.clip(q, Q_LOWER, Q_UPPER)


def validate_piper_kinematics() -> list[str]:
    errors: list[str] = []
    kin = PiperKinematics()
    zero_pos = kin.fk_pos(np.zeros(6, dtype=np.float64))
    if np.linalg.norm(zero_pos - EXPECTED_ZERO_FK_POS) > 1e-5:
        errors.append(f"zero FK mismatch: {zero_pos} vs {EXPECTED_ZERO_FK_POS}")
    if np.any(Q_LOWER >= Q_UPPER):
        errors.append("joint limits invalid")
    return errors
