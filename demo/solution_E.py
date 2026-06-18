"""Task E (Piper 桌面操作) —— 机械臂点到点精准控制方案。

本文件实现 AgileX Piper 机械臂的：
  1. 正运动学 (Forward Kinematics, FK)  —— 由关节角求末端 (gripper_base) 位姿；
  2. 逆运动学 (Inverse Kinematics, IK)  —— 由目标末端位姿求关节角（阻尼最小二乘 DLS）；
  3. 坐标系转换  —— 世界系 <-> 机械臂 base 系；
  4. （可选）RGB-D 反投影  —— 把相机深度像素恢复为三维世界坐标，供以后“按图找点”使用。

当前流程：**不做物体识别**，只需要你给定一个三维目标点，机械臂就会精准移动到该点。
用法见文件末尾 AlgSolution.set_target() 与 self.target_world 注释。

=====================================================================
                       坐标系总览（务必先读）
=====================================================================
本任务涉及 3 套坐标系，单位均为米 (m)，角度均为弧度 (rad)，旋转遵循右手定则：

(1) 世界系 (world)  —— Isaac Sim 全局坐标系，原点在环境原点 (0,0,0)，Z 轴向上。
    桌面、篮筐、相机、物体的位置都用世界系描述。是你“指定目标点”最直观的坐标系。
      · 桌面中心 (1.00, 0.00, 0.00)，桌面顶面高度 z ≈ 0.827。
      · 篮筐放置点 (1.08, -0.30, 0.977)。

(2) 机械臂 base 系 (base)  —— 固连在机械臂底座 base_link 上。
    正/逆运动学 (FK/IK) 全部在该系下计算（末端 gripper_base 相对 base 的位姿）。
    base 系相对世界系是一个**固定**的刚体变换（机械臂底座被固定在桌角）：
      · 平移 BASE_POS_W = (1.4043, 0, 0.827)        # base 原点在世界系中的坐标
      · 旋转 BASE_ROT_W = Rz(180°)                   # base 坐标轴相对世界轴绕 Z 转 180°
    180° 的朝向使机械臂“面向”桌子（+X_base 指向桌面中心方向）。

(3) 相机系 (camera)  —— 外部俯视相机 video_cam，固定在世界系中某处。
    仅在“RGB-D 反投影”里用到，把像素+深度恢复为世界系三维点。

转换关系（点的坐标变换，注意是“点”不是“向量”，要带平移）：
    p_world = BASE_POS_W + BASE_ROT_W · p_base          # base -> world
    p_base  = BASE_ROT_Wᵀ · (p_world - BASE_POS_W)      # world -> base
因 Rz(180°) 是对称正交矩阵，故 BASE_ROT_Wᵀ == BASE_ROT_W（见 world_to_base）。

数据流：
    你给的世界系目标点 ──world_to_base──▶ base 系目标点 ──IK──▶ 关节角 ──缩放──▶ 环境动作
"""

import numpy as np
import torch
from typing import Any


# =====================================================================
# 0. 常量（与 source/.../tasks/task_e/env_cfg.py 保持一致）
# =====================================================================

# 8 个关节的默认角（= 环境 init_state.joint_pos，也是 joint_pos_rel 的基准）
# 顺序: joint1..joint6 (臂) + joint7, joint8 (夹爪)
DEFAULT_JOINT_POS = np.array(
    [0.0, 1.2, -1.5, 0.0, 1.2, 0.0, 0.035, -0.035], dtype=np.float64
)

# 动作缩放: env 中 JointPositionActionCfg(scale=0.5, use_default_offset=True)
#   仿真实际下发的关节目标 = default + 0.5 * action
#   => 想让关节到达绝对角 q，则 action = (q - default) / 0.5
ACTION_SCALE = 0.5

# 夹爪开 / 合 (joint7, joint8)
GRIPPER_OPEN = np.array([0.035, -0.035], dtype=np.float64)
GRIPPER_CLOSE = np.array([-0.015, 0.015], dtype=np.float64)

# ---- 桌面 / 篮筐几何（由 env_cfg.py 推导，table scale=0.01）----
_TABLE_FACTOR = 0.01 / 0.008
_TABLE_DIMS = np.array(
    [0.6468062441005529, 0.9084968693231588, 0.6613141183247961]
) * _TABLE_FACTOR
TABLE_CENTER = np.array([1.00, 0.00, 0.00])
TABLE_HALF_X = _TABLE_DIMS[0] * 0.5             # ≈ 0.4043
TABLE_HALF_Y = _TABLE_DIMS[1] * 0.5             # ≈ 0.5678
TABLE_TOP_Z = TABLE_CENTER[2] + _TABLE_DIMS[2]  # ≈ 0.8266（桌面顶面世界高度）

BASKET_CENTER = np.array([1.08, -0.30, TABLE_TOP_Z + 0.15])  # 篮筐放置点(世界系)

# ---- 机械臂底座 base_link 在世界系下的位姿（固定，由 TaskEEnvPiperCfg 设定）----
#   init pos = (TABLE_CENTER_X + TABLE_HALF_X, 0, TABLE_TOP_Z)
#   init rot(w,x,y,z) = (0,0,0,1) -> 绕世界 Z 轴 180°，使机械臂朝向桌子
BASE_POS_W = np.array([TABLE_CENTER[0] + TABLE_HALF_X, 0.0, TABLE_TOP_Z])
BASE_ROT_W = np.array([            # Rz(180°)：base 坐标轴在世界系中的方向
    [-1.0, 0.0, 0.0],             #   +X_base 指向世界 -X（即指向桌面中心）
    [0.0, -1.0, 0.0],             #   +Y_base 指向世界 -Y
    [0.0, 0.0, 1.0],              #   +Z_base 仍与世界 +Z 同向（都向上）
])

# ---- 外部俯视 RGB-D 相机 video_cam 内外参（仅反投影用）----
CAM_H, CAM_W = 480, 640
_CAM_FOCAL, _CAM_APERTURE = 24.0, 20.955          # 针孔模型参数
CAM_FX = _CAM_FOCAL / _CAM_APERTURE * CAM_W       # ≈ 733.0
CAM_FY = CAM_FX                                   # 像素为正方形 -> fx == fy
CAM_CX, CAM_CY = CAM_W / 2.0, CAM_H / 2.0
CAM_POS_W = np.array([TABLE_CENTER[0] - 1.2, 0.0, TABLE_TOP_Z + 0.8])  # (-0.2,0,1.63)
CAM_ROT_W = np.array([0.957, 0.0, 0.290, 0.0])    # (w,x,y,z)，绕 Y 轴约 33.7°


# =====================================================================
# 1. 基础数学工具
# =====================================================================

def quat_to_mat(q):
    """四元数 (w, x, y, z) -> 3x3 旋转矩阵（USD/Isaac 的 scalar-first 约定）。"""
    w, x, y, z = q
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _T(pos, quat):
    """由平移 + 四元数构造 4x4 齐次变换矩阵。"""
    M = np.eye(4)
    M[:3, :3] = quat_to_mat(quat)
    M[:3, 3] = pos
    return M


def _Rz(a):
    """绕 Z 轴转 a 弧度的 4x4 齐次矩阵（Piper 所有转动关节轴均为本体 Z 轴）。"""
    c, s = np.cos(a), np.sin(a)
    M = np.eye(4)
    M[0, 0], M[0, 1] = c, -s
    M[1, 0], M[1, 1] = s, c
    return M


def _rot_error(R_cur, R_des):
    """两个旋转矩阵之间的姿态误差向量（IK 用，李代数 so(3) 近似）。

    误差 ≈ vee(R_des · R_curᵀ - I) 的反对称部分，量纲为弧度。
    """
    Re = R_des @ R_cur.T
    return 0.5 * np.array([
        Re[2, 1] - Re[1, 2],
        Re[0, 2] - Re[2, 0],
        Re[1, 0] - Re[0, 1],
    ])



def read_joint_state(obs):
    proprio = obs["proprio"]

    if hasattr(proprio, "detach"):
        proprio = proprio.detach().cpu().numpy()

    if proprio.ndim == 2:
        proprio = proprio[0]

    joint_pos_rel = proprio[0:8].astype(np.float32)
    joint_vel = proprio[8:16].astype(np.float32)
    last_actions = proprio[16:24].astype(np.float32)

    default_joint_pos = np.array(
        [0.0, 1.2, -1.5, 0.0, 1.2, 0.0, 0.035, -0.035],
        dtype=np.float32,
    )

    joint_pos_abs = default_joint_pos + joint_pos_rel

    return joint_pos_rel, joint_pos_abs, joint_vel, last_actions

# =====================================================================
# 2. 坐标系转换：世界系 <-> base 系
# =====================================================================
# 机械臂底座固定不动，因此 world<->base 是一个常量刚体变换。
# 记 base 系原点在世界系中的坐标为 t = BASE_POS_W，
#     base 系姿态（坐标轴方向）在世界系中的旋转矩阵为 R = BASE_ROT_W。
# 则对“点” p 有：
#     p_world = R · p_base + t          （把 base 系下的点表达到世界系）
#     p_base  = Rᵀ · (p_world - t)      （把世界系下的点表达到 base 系）
# 对“方向/向量” v（不含平移）则只乘旋转：v_world = R·v_base, v_base = Rᵀ·v_world。
# ---------------------------------------------------------------------

def world_to_base(p_w):
    """世界系坐标点 -> base 系坐标点（你给的目标点先经这一步再做 IK）。"""
    # Rz(180°) 满足 Rᵀ == R，所以这里直接用 BASE_ROT_W 即可（等价于 BASE_ROT_Wᵀ）。
    return BASE_ROT_W.T @ (np.asarray(p_w, dtype=np.float64) - BASE_POS_W)


def base_to_world(p_b):
    """base 系坐标点 -> 世界系坐标点（用于把 FK 结果转回世界系做校验/可视化）。"""
    return BASE_ROT_W @ np.asarray(p_b, dtype=np.float64) + BASE_POS_W


# =====================================================================
# 3. Piper 机械臂运动学（正解 FK + 逆解 IK）
# =====================================================================

class PiperKinematics:
    """Piper 6-DoF 手臂的正/逆运动学，全部在 base 系下计算。

    运动学链路 base_link -> gripper_base，每段相邻连杆变换为：
        T_i = Trans(localPos0_i) · Rot(localRot0_i) · Rz(q_i)
    其中 (localPos0_i, localRot0_i) 直接取自 piper.usd 中各关节定义，
    Rz(q_i) 为该关节绕本体 Z 轴转动 q_i。末端固定连杆 joint6_to_gripper_base
    为单位阵，可忽略。

    校验：q=0 时 FK 给出 gripper_base 在 base 系下位于 (0.0561, 0, 0.2132)，
    与 piper.usd 记录的静止位姿完全一致。
    """

    # (localPos0, localRot0(w,x,y,z))，逐关节取自 piper.usd
    _JOINTS = [
        ((0.0, 0.0, 0.123), (1.0, 0.0, 0.0, 0.0)),                                  # joint1
        ((0.0, 0.0, 0.0), (0.04800847, -0.048013408, -0.7054761, -0.7054739)),     # joint2
        ((0.28503, 0.0, 0.0), (0.6239962, 0.0, 0.0, -0.7814274)),                  # joint3
        ((-0.021984, -0.25075, 0.0), (0.7071055, 0.7071081, 0.0, 0.0)),            # joint4
        ((0.0, 0.0, 0.0), (0.7071055, -0.7071081, 0.0, 0.0)),                      # joint5
        ((0.000088259, -0.091, 0.0), (0.7071055, 0.7071081, 0.0, 0.0)),           # joint6
    ]

    # 各关节角限位（弧度，取自 USD lowerLimit/upperLimit）
    _Q_LOWER = np.deg2rad([-150.0, 0.0, -170.0, -100.0, -69.9, -120.0])
    _Q_UPPER = np.deg2rad([124.2, 179.9, 0.0, 100.0, 69.9, 120.0])

    # 预计算各段常量变换 (Trans·Rot)，IK 迭代时只需再乘 Rz(q)，省去重复构造
    _FIXED = [_T(p, r) for p, r in _JOINTS]

    # ---------------- 正解 ---------------- #
    def fk(self, q):
        """正运动学：6 个关节角 -> gripper_base 在 base 系下的 4x4 位姿矩阵。

        返回 M：M[:3,3] 为末端位置(base系)，M[:3,:3] 为末端姿态(base系)。
        """
        M = np.eye(4)
        for i in range(6):
            M = M @ self._FIXED[i] @ _Rz(q[i])
        return M

    def fk_pos(self, q):
        """正运动学便捷接口：只返回末端位置 (3,)（base 系）。"""
        return self.fk(q)[:3, 3]

    # ---------------- 雅可比 ---------------- #
    def jacobian(self, q, eps=1e-6):
        """数值雅可比 (6x6)：上 3 行为位置对各关节角的导数，下 3 行为姿态导数。"""
        M0 = self.fk(q)
        p0, R0 = M0[:3, 3], M0[:3, :3]
        J = np.zeros((6, 6))
        for j in range(6):
            dq = q.copy()
            dq[j] += eps
            Mj = self.fk(dq)
            J[:3, j] = (Mj[:3, 3] - p0) / eps              # 位置导数
            J[3:, j] = _rot_error(R0, Mj[:3, :3]) / eps    # 姿态导数
        return J

    # ---------------- 逆解 ---------------- #
    def ik(self, target_pos, target_R=None, q_init=None,
           iters=250, lam=0.05, ko=0.5, polish=40, step_clip=0.2):
        """逆运动学，在 base 系下求解，**位置严格优先**（亚毫米级精准）。

        核心思想：分层（任务优先级）求解 —— 把“到达目标位置”当作第一优先级任务，
        把“末端朝向目标姿态”当作第二优先级任务，且第二任务只能在**不破坏**第一任务
        的前提下（即在位置任务的零空间内）尽力完成。

        为什么不用简单的加权 DLS？
            若把位置、姿态误差一起塞进一个加权最小二乘，二者会通过同一个被阻尼的
            广义逆相互“串扰”。当目标姿态在该点本就不可达（典型：手腕 joint6 撞到
            ±120° 限位，无法完全俯视）时，求解器为了减小姿态误差会牺牲位置精度，
            实测残差可达数十毫米——这对“精准点到点”是不可接受的。

        分层解法（每次迭代）：
            ep = 位置误差(3)，Jp = 位置雅可比(3x6)
            位置任务步:   dq_pos = Jp# · ep            （Jp# 为阻尼伪逆）
            零空间投影:   N = I - Jp# · Jp             （把姿态步投到不影响位置的子空间）
            姿态任务步:   dq_ori = N · (ko · Joᵀ · eo) （只用零空间里剩余自由度调姿态）
            dq = dq_pos + dq_ori
        最后再做 `polish` 次**纯位置** DLS 迭代，把位置误差压到极致（亚毫米）。

        这样：位置一定精准到达；姿态是“能调多少调多少”的次要目标，
        joint6 限位饱和时也不会再连累位置。若 target_R=None 则只解位置。

        Parameters
        ----------
        target_pos : (3,)  base 系下的目标位置（必到达项，亚毫米精度）
        target_R   : (3,3) base 系下的目标姿态矩阵；None 时只约束位置
        q_init     : (6,)  迭代初值（务必传当前关节角，保证解连续、收敛快）
        lam        : 阻尼系数（越大越稳/越平滑，奇异点附近更安全）
        ko         : 姿态任务增益（零空间内调姿态的步长系数，越大越贴近目标姿态）
        polish     : 末尾纯位置精修迭代次数（把位置残差压到亚毫米）
        step_clip  : 单步关节更新上限（弧度），防迭代发散

        Returns
        -------
        q : (6,) 关节角解（已按关节限位裁剪）
        """
        q = (np.array(q_init, dtype=np.float64).copy()
             if q_init is not None else np.zeros(6))
        use_ori = target_R is not None
        I6 = np.eye(6)
        tgt = np.asarray(target_pos, dtype=np.float64)

        for _ in range(iters):
            M = self.fk(q)
            ep = tgt - M[:3, 3]                       # 位置误差(3,)
            J = self.jacobian(q)
            Jp = J[:3]                                # 位置雅可比(3x6)

            # 位置任务：阻尼伪逆 Jp# = Jpᵀ (Jp Jpᵀ + λ²I)⁻¹
            JJt = Jp @ Jp.T
            Jp_sharp = Jp.T @ np.linalg.inv(JJt + (lam ** 2) * np.eye(3))
            dq = Jp_sharp @ ep

            if use_ori:
                # 零空间投影矩阵 N：N·x 落在“不改变末端位置”的关节运动子空间里
                N = I6 - Jp_sharp @ Jp
                eo = _rot_error(M[:3, :3], target_R)  # 姿态误差(3,)
                Jo = J[3:]                            # 姿态雅可比(3x6)
                dq = dq + N @ (ko * (Jo.T @ eo))       # 仅在零空间内尽力调姿态

            q = q + np.clip(dq, -step_clip, step_clip)
            q = np.clip(q, self._Q_LOWER, self._Q_UPPER)

        # 末尾纯位置精修：只追位置，进一步把残差压到亚毫米级
        for _ in range(polish):
            M = self.fk(q)
            ep = tgt - M[:3, 3]
            if np.linalg.norm(ep) < 1e-7:
                break
            Jp = self.jacobian(q)[:3]
            JJt = Jp @ Jp.T
            dq = Jp.T @ np.linalg.solve(JJt + (lam ** 2) * np.eye(3), ep)
            q = q + np.clip(dq, -step_clip, step_clip)
            q = np.clip(q, self._Q_LOWER, self._Q_UPPER)

        return np.clip(q, self._Q_LOWER, self._Q_UPPER)


# =====================================================================
# 4. （可选）RGB-D 反投影：相机像素 -> 世界系三维点
#    当前流程不用识别；保留此工具，方便以后“在深度图上取一点 -> 得到世界坐标”。
# =====================================================================

def pixel_to_world(u, v, depth_value):
    """把外部相机的一个像素 (u行? 否，u为列, v为行) + 该处深度 -> 世界系三维点。

    采用 IsaacLab "world" 相机约定：相机本体系 +X 朝前(光轴)、+Y 朝左、+Z 朝上；
    深度为沿光轴的 z 距离 (distance_to_image_plane)。

    注意：相机内外参（CAM_*）来自配置文件推导，若实跑发现定位有偏差，
    优先校核这里的 CAM_ROT_W 与下面的光学系映射 R_cw_opt。
    """
    # 1) 像素 + 深度 -> 光学系(右,下,前)下的三维点
    x = (u - CAM_CX) * depth_value / CAM_FX
    y = (v - CAM_CY) * depth_value / CAM_FY
    z = depth_value
    p_opt = np.array([x, y, z])

    # 2) 光学系 -> 相机本体("world"约定)系：x_right=-Y, y_down=-Z, z_fwd=+X
    R_cw_opt = np.array([[0.0, 0.0, 1.0],
                         [-1.0, 0.0, 0.0],
                         [0.0, -1.0, 0.0]])
    # 3) 相机本体系 -> 世界系（外参）
    R_w_cam = quat_to_mat(CAM_ROT_W)
    return CAM_POS_W + R_w_cam @ (R_cw_opt @ p_opt)


# =====================================================================
# 5. 求解算法主体：给定三维点 -> 精准到达
# =====================================================================

class AlgSolution:
    """点到点精准控制。

    使用方法（二选一）：
      A. 直接改 self.target_world：在世界系下写死/设定一个目标点。
      B. 运行中调用 self.set_target(point, frame="world"/"base")：随时切换目标点。
    机械臂会用 IK 解出该点对应的关节角，并以限速方式精准移动过去后保持静止。
    """

    # 末端默认抓取姿态（base 系，俯视向下）：gripper 局部 Z 轴朝下
    #   对应四元数(w,x,y,z)=(0,1,0,0) -> 旋转矩阵 diag(1,-1,-1)
    GRASP_R = quat_to_mat([0.0, 1.0, 0.0, 0.0])


    def __init__(self):
        self.kin = PiperKinematics()

        # ============ 在这里设置你的目标点（世界坐标系，单位 m）============
        # 默认：桌面中心上方 15cm 处。改成你想去的任意点即可。
        #self.target_world = np.array([1.00, 0.00, TABLE_TOP_Z + 0.15])
        self.target_world = np.array([-0.2, -0.30, TABLE_TOP_Z + 0.65])

        # 末端姿态约束：
        #   self.GRASP_R -> 末端保持俯视向下（适合抓取）
        #   None         -> 只约束位置，不约束姿态（更容易到达，但末端朝向不定）
        self.target_orientation = self.GRASP_R

        # 夹爪状态（默认张开）。需要时改成 GRIPPER_CLOSE。
        self.gripper = GRIPPER_OPEN.copy()

        # 每个控制步关节最大变化量（弧度）。越小越平稳、越不易抖动，但到位越慢。
        self.max_joint_delta = 0.04

        # 运行时缓存（便于 debug：当前末端在世界系下的位置、到目标的距离）
        self.last_ee_world = None
        self.last_dist = None

    # ------------------------------------------------------------------ #
    def get_action_spec(self) -> dict[str, dict[str, Any]] | None:
        # 返回 None：使用官方默认动作配置（arm 位置控制, scale=0.5）
        return None

    # ------------------------------------------------------------------ #
    def set_target(self, point, frame: str = "world", orientation="grasp"):
        """设置目标点。

        point       : 长度 3 的三维点
        frame       : "world" 表示传入的是世界系坐标（默认）；"base" 表示已是 base 系坐标
        orientation : "grasp" 俯视向下抓取姿态；"free" 仅约束位置；或直接传 3x3 矩阵
        """
        p = np.asarray(point, dtype=np.float64).reshape(3)
        # 统一转成世界系存储，predicts 内部再转 base 系
        self.target_world = p if frame == "world" else base_to_world(p)
        if orientation == "grasp":
            self.target_orientation = self.GRASP_R
        elif orientation == "free":
            self.target_orientation = None
        else:
            self.target_orientation = np.asarray(orientation)

    # ------------------------------------------------------------------ #
    def _parse_qpos(self, obs):
        """从 obs 中取出当前 8 个关节的绝对角。

        proprio = [joint_pos_rel(8), joint_vel_rel(8), last_action(8)]，共 24 维，
        其中 joint_pos_rel = 当前角 - 默认角，所以绝对角 = joint_pos_rel + default。
        """
        proprio = obs["proprio"]
        if isinstance(proprio, torch.Tensor):
            proprio = proprio.detach().cpu().numpy()
        proprio = np.asarray(proprio).reshape(-1)
        return proprio[:8] + DEFAULT_JOINT_POS

    # ------------------------------------------------------------------ #
    def predicts(self, obs, current_score):
        # 1) 读取当前关节角（前 6 个是手臂）
        qpos = self._parse_qpos(obs)
        q_arm_cur = qpos[:6]

        # 2) 目标点：世界系 -> base 系（坐标系转换，见文件顶部说明）
        target_base = world_to_base(self.target_world)
        target_base = (0.1, -0.3, 0.4)

        # 3) 逆解：base 系目标位姿 -> 手臂关节角（以当前角为初值，保证连续收敛）
        q_arm_des = self.kin.ik(
            target_base, self.target_orientation, q_init=q_arm_cur
        )

        # 4) 限速：每步只朝目标角前进 max_joint_delta，避免大跳变导致抖动/过冲。
        #    稳态时 q_arm_des 不变，当前角会逐步收敛到它，FK 保证此时末端正好在目标点。
        delta = np.clip(q_arm_des - q_arm_cur,
                        -self.max_joint_delta, self.max_joint_delta)
        q_arm_cmd = q_arm_cur + delta

        # 5) 组装 8 维关节目标（臂 6 + 夹爪 2），再换算成环境动作
        full_target = np.concatenate([q_arm_cmd, self.gripper])
        action = (full_target - DEFAULT_JOINT_POS) / ACTION_SCALE

        # 6) debug 信息：当前末端在世界系下的位置 & 到目标的距离
        self.last_ee_world = base_to_world(self.kin.fk_pos(q_arm_cur))
        self.last_dist = float(np.linalg.norm(self.last_ee_world - self.target_world))
        joint_pos_rel, joint_pos_abs, joint_vel, last_actions = read_joint_state(obs)


        return {"action": action.tolist(), "giveup": False}
