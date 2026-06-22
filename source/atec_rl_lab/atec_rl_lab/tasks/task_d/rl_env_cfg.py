# Task-D 的 RSL-RL 训练配置：固定使用比赛任务 D 地图，训练 B2 采用“箱子垫高上平台”策略。


from isaaclab.envs import mdp  # 引入 IsaacLab 内置 MDP 函数，例如本体速度、关节状态、动作平滑惩罚等。
from isaaclab.managers import ObservationGroupCfg as ObsGroup  # 观测组配置基类，RSL-RL 需要 policy/critic 这样的观测组。
from isaaclab.managers import ObservationTermCfg as ObsTerm  # 单个观测项配置，例如关节位置、雷达高度扫描。
from isaaclab.managers import RewardTermCfg as RewTerm  # 单个奖励项配置，用来绑定 reward 函数、参数和权重。
from isaaclab.managers import EventTermCfg as EventTerm  # reset event 配置，用来按训练阶段重置 robot/box 初始状态。
from isaaclab.managers import TerminationTermCfg as DoneTerm  # termination 配置，用于 push 阶段箱子到位即结束。
from isaaclab.managers import SceneEntityCfg  # 场景实体选择器，用来指定 robot、box、lidar_sensor 以及关节/刚体名字。
from isaaclab.utils import configclass  # IsaacLab 配置类装饰器，支持嵌套配置和 __post_init__ 合并。
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise  # 均匀噪声配置，用于训练时模拟传感器误差。

import atec_rl_lab.tasks.task_d.mdp as task_d_mdp  # 引入任务 D 自定义奖励和终止函数。
from atec_rl_lab.tasks.task_d import constants as task_d_constants  # 共享 Task D 固定地图局部坐标。
from atec_rl_lab.tasks.task_d.env_cfg import TaskDEnvB2Cfg  # 继承官方任务 D 的 B2 比赛环境。


B2_LEG_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]


def sync_task_d_terrain_grid(env_cfg, min_env_spacing: float = 12.5):
    """Keep Task-D terrain tiles aligned with the actual number of parallel environments."""
    if not hasattr(env_cfg, "task_d_stage"):
        return
    terrain = getattr(env_cfg.scene, "terrain", None)
    terrain_generator = getattr(terrain, "terrain_generator", None)
    if terrain_generator is None:
        return
    env_cfg.scene.env_spacing = max(float(getattr(env_cfg.scene, "env_spacing", 0.0)), float(min_env_spacing))
    num_envs = int(env_cfg.scene.num_envs)
    terrain_generator.num_cols = num_envs
    terrain_generator.num_rows = 1
    terrain_generator.curriculum = False


@configclass  # 声明这是 IsaacLab 配置类，内部字段会被 Hydra/IsaacLab 正确解析。
class TaskDRLObservationsCfg:  # RSL-RL 使用的观测配置，替换比赛环境原本的 proprio/extero/image 结构。
    """RSL-RL observation groups using competition-available proprioception and lidar."""  # 说明 actor 只使用比赛可获得的本体和雷达信息。

    @configclass  # policy 观测组配置类，给 actor 使用。
    class PolicyCfg(ObsGroup):  # actor 的输入，不放箱子真实坐标或地形真值，避免使用比赛不可直接获得的信息。
        base_ang_vel = ObsTerm(  # 观测机器人本体角速度，用于保持姿态稳定。
            func=mdp.base_ang_vel,  # 调用 IsaacLab 内置函数读取 base 坐标系角速度。
            noise=Unoise(n_min=-0.1, n_max=0.1),  # 给角速度加入小噪声，提高策略鲁棒性。
            scale=0.25,  # 缩放角速度量纲，避免数值过大影响网络训练。
        )
        projected_gravity = ObsTerm(  # 观测重力在机体坐标系下的投影，用于感知 roll/pitch 姿态。
            func=mdp.projected_gravity,  # 读取 projected gravity，常用于腿式机器人姿态控制。
            noise=Unoise(n_min=-0.02, n_max=0.02),  # 加很小噪声模拟 IMU 姿态估计误差。
        )
        joint_pos = ObsTerm(  # 观测所有受控关节相对默认姿态的位置。
            func=mdp.joint_pos_rel,  # 使用相对关节位置，而不是绝对关节角，利于不同初始姿态泛化。
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},  # 先占位选择全部关节，后面 __post_init__ 会改成 B2 的实际 joint_names。
            noise=Unoise(n_min=-0.01, n_max=0.01),  # 给关节位置加入小噪声，模拟编码器误差。
        )
        joint_vel = ObsTerm(  # 观测所有受控关节速度。
            func=mdp.joint_vel_rel,  # 使用相对关节速度观测项。
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},  # 先占位选择全部关节，后面绑定实际 B2 关节顺序。
            noise=Unoise(n_min=-0.5, n_max=0.5),  # 关节速度噪声略大，因为速度估计通常比位置更 noisy。
            scale=0.05,  # 缩小关节速度数值，防止速度项主导网络输入。
        )
        actions = ObsTerm(func=mdp.last_action)  # 观测上一帧动作，让策略知道自身控制历史，减少动作抖动。
        lidar_scan = ObsTerm(  # 观测比赛提供的激光/高度扫描后处理结果，用来看到坑、平台和箱子。
            func=mdp.height_scan,  # 使用 IsaacLab height_scan 从 lidar_sensor 的 ray hit 转成高度差特征。
            params={"sensor_cfg": SceneEntityCfg("lidar_sensor")},  # 指定使用任务 D 里已经包含 ground+box 的 MultiMesh lidar_sensor。
            clip=(-2.0, 2.0),  # 裁剪高度扫描值，避免极端 ray miss 或深坑数值破坏训练。
            scale=1.0,  # 保持高度扫描原始尺度，便于策略判断障碍高度。
        )

        def __post_init__(self):  # 观测组初始化后配置。
            self.enable_corruption = False  # 关闭 IsaacLab 统一观测扰动；上面已经对关键项单独加噪声。
            self.concatenate_terms = True  # 将所有 policy 观测项拼成一个向量，符合 RSL-RL MLP 输入格式。

    @configclass  # critic 观测组配置类，给价值函数使用。
    class CriticCfg(ObsGroup):  # critic 可以比 actor 多看 base_lin_vel，帮助估值稳定，但仍不直接给箱子真值。
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0)  # critic 观测本体线速度，提高 value 对运动状态的判断能力。
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)  # critic 观测角速度，判断是否快要失稳。
        projected_gravity = ObsTerm(func=mdp.projected_gravity)  # critic 观测姿态，用于估计摔倒/爬升状态价值。
        joint_pos = ObsTerm(  # critic 观测关节位置。
            func=mdp.joint_pos_rel,  # 使用相对关节位置。
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},  # 后面替换成实际 joint_names。
        )
        joint_vel = ObsTerm(  # critic 观测关节速度。
            func=mdp.joint_vel_rel,  # 使用相对关节速度。
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},  # 后面替换成实际 joint_names。
            scale=0.05,  # 缩放速度，保持输入尺度稳定。
        )
        actions = ObsTerm(func=mdp.last_action)  # critic 也看上一帧动作，便于估计动作惯性和抖动惩罚。
        lidar_scan = ObsTerm(  # critic 同样使用 lidar_scan，保持与 actor 的环境几何感知一致。
            func=mdp.height_scan,  # 从 lidar_sensor 计算高度扫描。
            params={"sensor_cfg": SceneEntityCfg("lidar_sensor")},  # 指定任务 D 的 lidar_sensor。
            clip=(-2.0, 2.0),  # 裁剪扫描值，避免异常高度影响 value 学习。
        )

        def __post_init__(self):  # critic 观测组初始化后配置。
            self.enable_corruption = False  # critic 不加随机扰动，使用干净状态估值更稳定。
            self.concatenate_terms = True  # 拼接成一个 critic 输入向量。

    policy: PolicyCfg = PolicyCfg()  # 注册 policy 观测组，RSL-RL 会把它作为 actor 输入。
    critic: CriticCfg = CriticCfg()  # 注册 critic 观测组，RSL-RL 会把它作为 critic 输入。


@configclass  # 声明奖励配置类。
class TaskDRLRewardsCfg:  # TaskD 专用训练奖励：比赛得分项 + 垫高上平台过程奖励 + 动作正则。
    achieve = RewTerm(  # 机器人沿 x 方向越过比赛得分线的奖励。
        func=task_d_mdp.RewardCrossX,  # 使用任务 D 原有的一次性跨线奖励函数。
        params={  # RewardCrossX 的参数字典。
            "asset_cfg": SceneEntityCfg("robot"),  # 读取 robot 的 root x 位置。
            "threshold": [1.6, 5.0],  # 两条跨线阈值：先过坑前关键点，再过更靠后的通过点。
            "reward_value": [10.0, 80.0],  # 训练中放大跨线奖励，引导最终通过任务 D。
            "debug": False,  # 训练时关闭逐步打印，避免日志过多。
            "visual_assets": False,  # 训练时不生成可视化阈值线，减少 headless 开销。
        },
        weight=1.0,  # 使用函数内部 reward_value 的尺度，不再额外缩放。
    )
    box_in_target_x = RewTerm(  # 箱子进入比赛目标 x 区间的奖励。
        func=task_d_mdp.RewardBoxXInRange,  # 使用任务 D 原有箱子 x 区间奖励函数。
        params={  # RewardBoxXInRange 的参数字典。
            "asset_cfg": SceneEntityCfg("box"),  # 读取 box 的 root x 位置。
            "x_min": [2.3, 1.6],  # 两个目标区间的左边界，保持与比赛任务 D 配置一致。
            "x_max": [3.7, 2.3],  # 两个目标区间的右边界，保持与比赛任务 D 配置一致。
            "reward_value": 60.0,  # 提高箱子到位奖励，让策略优先学会利用箱子。
            "one_time": True,  # 每个 episode 只给一次箱子到位奖励，避免原地刷分。
            "debug": False,  # 关闭调试打印。
        },
        weight=1.0,  # 不额外缩放箱子到位奖励。
    )

    box_progress = RewTerm(
        func=task_d_mdp.BoxForwardProgress,
        params={"start_x": task_d_constants.FULL_BOX_START[0], "target_x": task_d_constants.CLIMB_BOX_TARGET_X},
        weight=30.0,
    )  # 增量奖励箱子从初始 local x=0 向垫高目标 local x=2 移动；停住不再刷分。
    box_step_target = RewTerm(  # 奖励箱子接近指定垫高位置。
        func=task_d_mdp.BoxToStepTargetProgress,  # 计算箱子到固定 target_xy 的指数型距离奖励。
        params={"target_xy": (task_d_constants.CLIMB_BOX_TARGET_X, task_d_constants.CLIMB_BOX_TARGET_Y), "std": 0.75},  # target_xy 指定箱子垫高位置；std 控制奖励衰减范围。
        weight=2.0,  # 低权重连续引导箱子靠近目标；主要奖励仍由增量推进和一次性阶段奖励承担。
    )
    robot_near_box = RewTerm(func=task_d_mdp.robot_near_box, weight=1.0)  # 小奖励机器人靠近箱子，帮助早期探索接触箱子。
    robot_on_box = RewTerm(  # 一次性奖励机器人踩到“已经推到目标附近”的箱子上，避免原地爬初始箱子刷分。
        func=task_d_mdp.RobotOnBoxOnce,
        params={"target_xy": (task_d_constants.CLIMB_BOX_TARGET_X, task_d_constants.CLIMB_BOX_TARGET_Y), "box_ready_radius": 0.25},
        weight=15.0,
    )
    robot_on_platform_side = RewTerm(  # 一次性奖励机器人到达高台/平台侧，避免停在平台边每步刷分。
        func=task_d_mdp.RobotOnPlatformSideOnce,
        params={"min_x": 2.8, "min_z": 1.35, "target_y": task_d_constants.CLIMB_BOX_TARGET_Y, "y_half_width": 1.4, "target_xy": (task_d_constants.CLIMB_BOX_TARGET_X, task_d_constants.CLIMB_BOX_TARGET_Y)},
        weight=30.0,
    )
    robot_progress = RewTerm(func=task_d_mdp.RobotForwardProgress, weight=20.0)  # 增量奖励机器人向终点 x=3.5 前进；停住不再刷分。
    alive_time_penalty = RewTerm(func=mdp.is_alive, weight=-0.01)  # 每步小时间惩罚，促使策略完成任务而不是停留。

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.02)  # 惩罚动作变化过快，避免策略抖动和仿真不稳定。
    joint_torques_l2 = RewTerm(  # 惩罚关节力矩过大，减少硬推、暴力动作和能耗。
        func=mdp.joint_torques_l2,  # IsaacLab 内置关节力矩平方惩罚。
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},  # 作用于 robot 的全部关节。
        weight=-1.0e-5,  # 较小负权重，避免压制推箱子和爬升所需力矩。
    )
    joint_acc_l2 = RewTerm(  # 惩罚关节加速度过大，让动作更平滑。
        func=mdp.joint_acc_l2,  # IsaacLab 内置关节加速度平方惩罚。
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},  # 作用于全部关节。
        weight=-1.0e-7,  # 很小负权重，只做正则，不阻碍上箱子动作。
    )
    joint_pos_soft_limit = RewTerm(  # 轻度惩罚腿关节靠近软限位，减少早期异常姿态。
        func=task_d_mdp.joint_pos_near_soft_limits,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=B2_LEG_JOINT_NAMES), "soft_margin": 0.20},
        weight=-0.5,
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5)  # 轻微惩罚身体倾斜；权重不大，允许爬箱子时有姿态变化。


@configclass  # 声明 TaskD RL 环境配置类。
class TaskDRLEnvB2Cfg(TaskDEnvB2Cfg):  # 继承官方 TaskD B2 环境，保留地图、箱子、终止和比赛基本逻辑。
    task_d_stage: str = "full"  # 训练阶段：climb/drop/push/mixed/full；不同阶段使用不同 reset 和奖励开关。
    mixed_stage_weights: dict[str, float] | None = None  # mixed reset 权重；None 时使用 events.py 默认权重。
    debug_reset_positions: bool = False  # 临时 reset 调试开关；打印 env origin/local/world pose 对照。
    debug_reset_num_envs: int = 4  # reset 调试时最多打印多少个 env。
    pit_width_range: tuple[float, float] = (1.35, 1.35)  # 背地图训练先固定坑宽，成功后再恢复官方范围。
    platform_height_range: tuple[float, float] = (1.10, 1.10)  # 背地图训练先固定平台高度，降低课程初期随机性。
    """Fixed-map Task-D training env for B2 and the box-as-step strategy."""  # 文档说明：固定地图，训练箱子垫高策略。

    rewards: TaskDRLRewardsCfg = TaskDRLRewardsCfg()  # 将官方稀疏比赛奖励替换为训练用密集奖励配置。

    def __post_init__(self):  # 环境配置初始化后处理。
        super().__post_init__()  # 先运行官方 TaskDEnvB2Cfg 初始化，生成比赛地形、箱子、B2 机器人和 lidar。
        from atec_rl_lab.assets.robots import UNITREE_B2_CFG

        self.scene.robot = UNITREE_B2_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=UNITREE_B2_CFG.init_state.replace(
                pos=(-3, 0.0, task_d_constants.B2_STANDING_ROOT_Z),
                joint_pos=task_d_constants.B2_STANDING_JOINT_POS,
                joint_vel=UNITREE_B2_CFG.init_state.joint_vel,
            ),
        )
        self.scene.robot.joint_names = B2_LEG_JOINT_NAMES
        self.scene.robot.leg_joint_names = B2_LEG_JOINT_NAMES
        self.scene.robot.arm_joint_names = []
        self.rewards = TaskDRLRewardsCfg()  # 父类 TaskDEnvCfg.__post_init__ 会写 self.rewards=RewardsCfg()，这里必须重新覆盖成 RL dense rewards。

        self.scene.num_envs = 1024  # 默认并行环境数；显存不够时可用 --num_envs 覆盖。
        sync_task_d_terrain_grid(self)  # Task D terrain footprint is 12m x 8m；每个 env 都需要独立地形 tile。
        self.events.reset_task_d_stage = EventTerm(  # 用自定义 reset event 固定/课程化 robot 和 box 初始状态。
            func=task_d_mdp.reset_task_d_stage,  # 调用 task_d/mdp/events.py 里的分阶段 reset 函数。
            mode="reset",  # 每次 episode reset 时执行。
            params={
                "stage": self.task_d_stage,
                "robot_default_z": task_d_constants.B2_STANDING_ROOT_Z,
                "box_default_z": 0.5,
                "mixed_stage_weights": self.mixed_stage_weights,
                "debug": self.debug_reset_positions,
                "debug_num_envs": self.debug_reset_num_envs,
            },  # 按当前 stage 设置 robot/box 位姿。
        )
        self.episode_length_s = 20.0  # 单回合最长 20 秒，任务 D 训练不需要原比赛 20 分钟超长 episode。
        self.scene.head_camera = None  # 关闭头部相机；RL 输入不用图像，headless 训练无需 --enable_cameras。
        self.scene.ee_camera = None  # 关闭末端相机；避免 B2 asset 自动生成相机导致 headless 报错。
        self.scene.ee_dual_camera = None  # 关闭双目末端相机；本任务只使用 lidar 和本体状态。
        self.observations = TaskDRLObservationsCfg()  # 父类初始化完相机/lidar 后，再替换成 RSL-RL 的 policy/critic 观测结构。

        joint_names = B2_LEG_JOINT_NAMES  # Task D RL 使用 plain B2，只保留 12 个腿部关节。
        leg_joint_names = B2_LEG_JOINT_NAMES  # 腿部 action 只控制 B2 腿关节。

        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = joint_names  # actor 的关节位置观测绑定到 B2 腿关节。
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = joint_names  # actor 的关节速度观测绑定到 B2 腿关节。
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = joint_names  # critic 的关节位置观测绑定到 B2 腿关节。
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = joint_names  # critic 的关节速度观测绑定到 B2 腿关节。

        self.actions.joint_leg.joint_names = leg_joint_names  # 腿部 action 只控制腿关节，继承 BaseEnv 的 position action 类型。
        self.actions.joint_arm = None  # Task D 固定地图策略不使用 Piper 机械臂，plain B2 action 维度为 12。
        self.actions.joint_wheel = None  # B2 没有轮子，显式关闭 wheel action。

        self.events.physics_material = None  # 关闭材质随机化，地图和摩擦保持比赛先验固定，便于先出结果。
        self.events.base_external_force_torque = None  # 关闭 reset 外力扰动，避免早期训练被无关扰动干扰。
        self.events.reset_robot_joints = None  # Task D RL 由 reset_task_d_stage 显式写入 B2 固定站姿，避免继承 reset 覆盖。
        self.terminations.fall.params["minimum_height"] = 0.25  # base 高度低于 0.25m 判定摔倒，允许上箱子过程中的高度变化。
        self.terminations.joint_pos_hard_limit = DoneTerm(
            func=task_d_mdp.JointPositionHardLimit,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=B2_LEG_JOINT_NAMES),
                "hard_margin": 0.20,
                "consecutive_frames": 8,
                "grace_steps": 10,
                "debug": self.debug_reset_positions,
                "debug_num_envs": self.debug_reset_num_envs,
            },
            time_out=False,
        )

        if self.task_d_stage == "climb":  # 第一阶段：箱子已到位，只训练上箱子/上平台/越线。
            self.rewards.box_progress = None  # climb 阶段不需要推箱子进度奖励。
            self.rewards.box_step_target = None  # 箱子已在目标附近，不再奖励箱子靠近目标。
            self.rewards.robot_near_box = None  # 机器人已从箱子前开始，不需要靠近箱子 shaping。
            self.rewards.box_in_target_x = None  # 暂时不优化比赛箱子得分，专注爬升动作。
        elif self.task_d_stage == "drop":  # 第二阶段：从平台边/平台上开始，训练下平台后继续冲线。
            self.rewards.achieve.params["threshold"] = [5.0]  # drop reset 已经在 local x>1.6 后方，避免开局白拿第一段奖励。
            self.rewards.achieve.params["reward_value"] = [80.0]  # drop 只保留终点侧跨线奖励。
            self.rewards.box_progress = None  # drop 阶段不推箱子。
            self.rewards.box_step_target = None  # drop 阶段箱子固定在目标附近。
            self.rewards.robot_near_box = None  # drop 阶段不需要靠近箱子。
            self.rewards.robot_on_box = None  # drop 阶段不训练上箱子。
            self.rewards.robot_on_platform_side = None  # reset 已经在平台侧，避免一开局拿阶段奖励。
            self.rewards.box_in_target_x = None  # drop 阶段不训练箱子得分。
        elif self.task_d_stage == "push":  # 第三阶段：只训练从原始位置推箱子到目标点。
            self.rewards.achieve = None  # push 阶段不奖励机器人越线，防止过早乱爬/乱冲。
            self.rewards.robot_on_box = None  # push 阶段不奖励上箱子。
            self.rewards.robot_on_platform_side = None  # push 阶段不奖励上平台。
            self.rewards.robot_progress = None  # push 阶段不奖励机器人自己前进，只奖励箱子推进。
            self.terminations.box_at_push_target = DoneTerm(  # Push 成功必须能接上 Climb reset：箱子到位、机器人在预爬入口且姿态对齐。
                func=task_d_mdp.push_ready_for_climb,
                params={
                    "robot_cfg": SceneEntityCfg("robot"),
                    "box_cfg": SceneEntityCfg("box"),
                    "box_target_xy": (task_d_constants.CLIMB_BOX_TARGET_X, task_d_constants.CLIMB_BOX_TARGET_Y),
                    "box_x_half_width": task_d_constants.CLIMB_BOX_HALF_WIDTH_X,
                    "box_y_half_width": task_d_constants.CLIMB_BOX_HALF_WIDTH_Y,
                    "robot_target_xy": (task_d_constants.PRE_CLIMB_ROBOT_X, task_d_constants.PRE_CLIMB_ROBOT_Y),
                    "robot_x_half_width": task_d_constants.PRE_CLIMB_ROBOT_HALF_WIDTH_X,
                    "robot_y_half_width": task_d_constants.PRE_CLIMB_ROBOT_HALF_WIDTH_Y,
                    "robot_max_abs_yaw": task_d_constants.PRE_CLIMB_YAW_RANGE[1],
                    "box_max_abs_yaw": task_d_constants.CLIMB_BOX_YAW_TOL,
                },
                time_out=False,
            )
        elif self.task_d_stage == "mixed":  # 混合课程：每个 env reset 时采样 push/climb/drop/full 起点。
            self.rewards.achieve.params["threshold"] = [5.0]  # mixed 含 drop reset，避免 drop 子样本开局拿 local x>1.6 奖励。
            self.rewards.achieve.params["reward_value"] = [80.0]  # 保留最终过平台/终点侧奖励。
            self.rewards.box_progress = None  # mixed 含 climb/drop reset，避免已到位箱子因抖动拿无意义进度奖励。
            self.rewards.box_step_target = None  # mixed 含 climb/drop reset，避免箱子已到位子样本每步白拿目标奖励。
            self.rewards.box_in_target_x = None  # mixed 含箱子已到位子样本，避免开局白拿箱子区间奖励。
            self.rewards.robot_near_box = None  # mixed 子样本目标不同，避免停在箱子附近刷 dense reward。
            self.rewards.robot_on_box = None  # mixed 含 climb/drop reset，避免重置状态直接或过早吃上箱奖励。
            self.rewards.robot_on_platform_side = None  # mixed 含 drop reset，避免平台侧起点白拿一次性奖励。
        elif self.task_d_stage == "full":  # 最终阶段：完整比赛起点，保留全部奖励。
            pass  # full 阶段不关闭奖励。
        else:  # 配置错误保护。
            raise ValueError(f"Unsupported Task-D RL stage: {self.task_d_stage}")  # 明确提示非法 stage。


@configclass
class TaskDRLEnvB2ClimbCfg(TaskDRLEnvB2Cfg):  # Climb 课程：箱子已到位，机器人从箱子前开始。
    task_d_stage: str = "climb"  # 只训练上箱子、上平台、继续越线。


@configclass
class TaskDRLEnvB2DropCfg(TaskDRLEnvB2Cfg):  # Drop 课程：机器人从平台边/平台上开始。
    task_d_stage: str = "drop"  # 只训练下平台后稳定行走并冲线。


@configclass
class TaskDRLEnvB2PushCfg(TaskDRLEnvB2Cfg):  # Push 课程：箱子原位，机器人从箱子附近开始。
    task_d_stage: str = "push"  # 只训练把箱子推到共享 Climb box target 附近。


@configclass
class TaskDRLEnvB2MixedCfg(TaskDRLEnvB2Cfg):  # Mixed 课程：按 env 随机采样 push/climb/drop/full reset。
    task_d_stage: str = "mixed"  # 默认权重在 reset event 中定义，可通过 mixed_stage_weights 覆盖。


@configclass
class TaskDRLEnvB2FullCfg(TaskDRLEnvB2Cfg):  # Full 课程：完整比赛起点。
    task_d_stage: str = "full"  # 串联推箱子、上箱子、上平台、冲线。
