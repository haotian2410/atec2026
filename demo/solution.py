"""TaskB 主策略入口。

本文件负责比赛主状态机、底盘速度指令、相机帧保存和 YOLO 结果轮询；
具体的 YOLO JSON 解析、深度反投影和视觉伺服公式在 demo/tool/yolo_targets.py。

当前垃圾搜索/接近流程：
1. GO_TO_DROP_STAND / FACE_DROP_CENTER：先到投放区外侧站位并面向投放区；
2. FACE_NEAR_SWEEP_START：转到背向投放区的有效搜索扇区起点，期间也保存 live 帧；
3. NEAR_SWEEP_TRASH：低速近扫，只允许 head/body 近目标或很近的 ee 目标接管；
4. FAR_SEARCH_TRASH：近处没目标后，末端相机顺时针扫约 165 度有效扇区，
   锁定一个 ee 远目标后进入粗接近，不继续看边界/投放区矮墙；
5. APPROACH_TRASH_TARGET：持续保存最新 live 帧并刷新 YOLO。ee 远目标保持锁定粗靠近，
   一旦 head/body 最新帧看到近目标，切到本体相机微调；
6. PRONE_TRANSITION：本体相机近距离微调稳定后，不再继续挪底盘，而是在动作层
   把 12 个腿关节从当前 RL 站姿动作线性插值到固定趴下姿态，降低机身和末端高度；
7. READY_TO_GRASP：趴下插值完成后保持固定趴姿，底盘静止，等待后续机械臂抓取逻辑接入。

趴下逻辑说明：
- 状态层只负责决定何时进入 PRONE_TRANSITION，并把底盘速度命令置零；
- 动作层仍先运行原来的 locomotion policy，得到当前帧的站立腿部动作；
- _apply_prone_override() 再按 alpha 将腿部动作平滑混到 fixed_prone_leg_action；
- alpha 在 PRONE_TRANSITION 中从 0 到 1，进入 READY_TO_GRASP 后固定为 1；
- 机械臂 8 维动作目前仍保持 0，不在这里做抓取。

关键函数位置：
- _compute_navigation_command(): 主状态机和每个阶段的速度决策；
- _refresh_yolo_targets(): 轮询 yolo_results.json，避免接近阶段用通用选择器破坏远端锁；
- _select_far_search_target() / _lock_or_choose_far_target(): ee 远搜目标选择和稳定锁定；
- _select_approach_target() / _choose_latest_body_target(): 接近阶段相机切换和最新 head/body 帧优先；
- _write_pose_state(): 给 pose_viewer 和调试用的状态快照；
- _prone_override_alpha() / _apply_prone_override(): 预抓取阶段的固定趴姿插值覆盖。

solution.py 只负责调度和控制闭环；地图先验、投放区拟合、YOLO 数据契约、
Piper 运动学等细节放在 demo/tool 目录，避免主状态机继续膨胀。
"""

import os
from typing import Any
import math
import time
import json
import subprocess
import threading
import numpy as np
import torch

try:
    from demo.tool.drop_zone import DropZoneTracker, estimate_from_obs, estimate_world_drop_from_obs
    from demo.tool.localizer2d import Localizer2D
    from demo.tool import taskb_map_prior as map_prior
    from demo.tool.piper_kinematics import validate_piper_kinematics
    from demo.tool.lidar_height_scan import estimate_drop_from_height_scan, debug_height_scan_projection
    from demo.tool.yolo_targets import load_yolo_targets, servo_command_from_target
    from demo.tool.taskb_yolo_runner import TaskBYoloRunner
except ImportError:
    from tool.drop_zone import DropZoneTracker, estimate_from_obs, estimate_world_drop_from_obs
    from tool.localizer2d import Localizer2D
    from tool import taskb_map_prior as map_prior
    from tool.piper_kinematics import validate_piper_kinematics
    from tool.lidar_height_scan import estimate_drop_from_height_scan, debug_height_scan_projection
    from tool.yolo_targets import load_yolo_targets, servo_command_from_target
    from tool.taskb_yolo_runner import TaskBYoloRunner

class AlgSolution:

    # 官方默认 leg/arm position action 都是：target = default_joint_pos + scale * action。
    # 这里保持默认 scale=0.5，因此固定姿态动作需要用
    # (目标绝对关节角 - 默认绝对关节角) / 0.5 反推出来。
    ACTION_SCALE = 0.5

    # Task 环境步长是 0.02s；趴下插值按 step_count 计时，避免依赖 wall clock。
    CONTROL_DT = 0.02
    PRONE_TRANSITION_SECONDS = 1.0

    # B2Piper 腿部 12 维关节顺序固定为：
    # FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf,
    # RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf。
    # 这些默认角来自 assets/robots/b2.py 里的 UNITREE_B2_PIPER_CFG.init_state。
    DEFAULT_LEG_POS = (
        -0.1, 0.8, -1.5,
        0.1, 0.8, -1.5,
        -0.1, 1.0, -1.5,
        0.1, 1.0, -1.5,
    )
    # 固定趴下姿态的绝对关节目标，单位是弧度。
    # 这组值来自前面单独测试过的固定趴姿：四条腿折叠后能降低机身，
    # 同时比“前腿跪姿”更不容易让头部先撞地。
    PRONE_LEG_TARGET = (
        -0.1, 1.55, -2.55,
        0.1, 1.55, -2.55,
        -0.1, 1.55, -2.55,
        0.1, 1.55, -2.55,
    )
    EE_BODY_NAME_CANDIDATES = ("gripper_base", "piper_gripper_base")
    ARM_JOINT_NAME_CANDIDATES = (
        ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        ["arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5", "arm_joint6"],
    )

    def __init__(self):
        policy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy.pt")
        # 比赛/仿真一般有 CUDA；本地梳理或单元检查时允许退回 CPU，避免直接加载失败。
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.policy = torch.jit.load(policy_path, map_location=self.device)
        self.policy.eval()

        # 保留这两个可选策略的加载，方便以后对比；当前预抓取趴下不使用它们。
        # 原因是这次要的是确定、可复现的固定趴姿：先由 locomotion policy 给出
        # 当前站立动作，再用 _apply_prone_override() 平滑覆盖腿部 12 维。
        self.prone_policy = self._load_optional_leg_policy("prone_policy.pt")
        self.low_stance_policy = self._load_optional_leg_policy("low_stance_policy.pt")
        self.active_leg_policy_name = "locomotion"

        self.leg_action_dim = 12
        self.arm_action_dim = 8

        self.leg_joint_indices = list(range(12))
        self.arm_joint_indices = list(range(12, 20))

        self.train_to_env_action_scale = torch.tensor(
            [
                0.25, 0.5, 0.5,
                0.25, 0.5, 0.5,
                0.25, 0.5, 0.5,
                0.25, 0.5, 0.5,
            ],
            device=self.device,
            dtype=torch.float32,
        ).view(1, -1)

        self.env_to_train_action_scale = torch.tensor(
            [
                4.0, 2.0, 2.0,
                4.0, 2.0, 2.0,
                4.0, 2.0, 2.0,
                4.0, 2.0, 2.0,
            ],
            device=self.device,
            dtype=torch.float32,
        ).view(1, -1)

        # 行走策略的速度指令槽位由状态机动态写入；初始化值只用于 reset 后的第一帧。
        self.fixed_velocity_commands = torch.tensor(
            #[0.5, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            device=self.device,
            dtype=torch.float32,
        ).view(1, 3)

        self.arm_default_action = torch.zeros(
            (1, self.arm_action_dim),
            device=self.device,
            dtype=torch.float32,
        )
        # 预先把“绝对趴姿关节角”换算成环境需要的 action。
        # predicts() 每帧只需要 repeat 和线性插值，不再重复做张量构造。
        default_leg_pos = torch.tensor(
            self.DEFAULT_LEG_POS,
            device=self.device,
            dtype=torch.float32,
        ).view(1, -1)
        prone_leg_target = torch.tensor(
            self.PRONE_LEG_TARGET,
            device=self.device,
            dtype=torch.float32,
        ).view(1, -1)
        self.fixed_prone_leg_action = (prone_leg_target - default_leg_pos) / self.ACTION_SCALE
        self.prone_transition_total_steps = max(
            1,
            int(round(self.PRONE_TRANSITION_SECONDS / self.CONTROL_DT)),
        )

        self.reset()

    def _load_optional_leg_policy(self, filename: str):
        policy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        if not os.path.exists(policy_path):
            return None
        try:
            policy = torch.jit.load(policy_path, map_location=self.device)
            policy.eval()
            return policy
        except Exception as err:
            print(f"[AlgSolution] failed to load optional leg policy {policy_path}: {err!r}")
            return None

    def reset(self, **kwargs):
        if getattr(self, "yolo_started", False):
            self._stop_yolo_runner()

        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.scan_dir = os.path.join(
            self.project_root,
            "outputs",
            "taskb_scan",
            f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}",
        )
        self.pose_state_path = os.path.join(self.project_root, "outputs", "taskb_pose_state.json")
        self.lidar_debug_dir = os.path.join(self.project_root, "outputs", "taskb_lidar_debug")
        self.lidar_debug_path = os.path.join(self.lidar_debug_dir, "latest_lidar_debug.npz")

        self.drop_tracker = DropZoneTracker()
        self.phase = "GO_TO_DROP_STAND"
        self.active_leg_policy_name = "locomotion"
        self.step_count = 0
        self.last_score = 0.0

        # 2D 定位器维护“策略层认为的机器人位姿”。
        # 第一版用出生点 + proprio 速度积分预测；地图、投放区、相机/LiDAR 参数
        # 都放在 demo/tool/taskb_map_prior.py 里作为先验。后续接入圆环/边界观测时，
        # 只需要在 Localizer2D 里做校正，不必再大改状态机。
        self.localizer = Localizer2D(dt=0.02)
        self.prior_errors = map_prior.validate_taskb_prior() + validate_piper_kinematics()
        self.last_lidar_diag = {"available": False, "reason": "not_checked"}
        self.last_visual_correction = None
        self.trash_targets = []
        self.current_trash_target = None
        self.current_trash_servo = None
        self.last_yolo_diag = {"available": False, "reason": "not_checked"}
        self.ready_to_grasp_steps = 0
        self.ready_to_grasp_last_image = None
        self.ready_to_grasp_last_target = None
        # PRONE_TRANSITION 的起始 step。为 None 表示当前还没开始趴下插值；
        # 一旦本体相机近距离微调确认 ready_to_grasp，会记录当前 step_count。
        self.prone_transition_start_step = None
        self.locked_trash_target = None
        self.locked_trash_miss_steps = 0
        self.far_locked_target = None
        self.far_locked_miss_steps = 0
        self.trash_servo_mode = "live_search"
        self.body_target_lost_steps = 0
        self.ee_target_lost_steps = 0
        self.approach_target_lost_steps = 0
        self.last_yolo_empty_step = -1
        self.live_frame_interval = 3
        self.last_live_frame_step = -1
        self.yolo_enabled = os.environ.get("ATEC_DISABLE_EMBEDDED_YOLO", "0") != "1"
        self.yolo_model_path = os.environ.get("TASKB_YOLO_MODEL")
        if not self.yolo_model_path:
            self.yolo_model_path = os.path.join(
                self.project_root,
                "models",
                "hf",
                "esapzoi_litter_yolov8",
                "best.pt",
            )
        self.yolo_conf = float(os.environ.get("TASKB_YOLO_CONF", "0.25"))
        self.yolo_watch_interval = float(os.environ.get("TASKB_YOLO_INTERVAL", "0.35"))
        self.yolo_max_live_per_camera = int(os.environ.get("TASKB_YOLO_MAX_LIVE", "1"))
        self.yolo_max_result_age_s = float(os.environ.get("TASKB_YOLO_MAX_AGE_S", "1.2"))
        self.yolo_max_live_lag_steps = int(os.environ.get("TASKB_YOLO_MAX_LIVE_LAG_STEPS", "15"))
        self.yolo_thread = None
        self.yolo_stop_event = None
        self.yolo_process = None
        self.yolo_log_handle = None
        self.yolo_started = False
        self.yolo_start_error = None
        self.yolo_stale_count = 0
        self.yolo_stale_restart_threshold = int(os.environ.get("TASKB_YOLO_STALE_RESTART_COUNT", "3"))
        self.yolo_restart_backoff_s = float(os.environ.get("TASKB_YOLO_RESTART_BACKOFF_S", "2.0"))
        self.last_yolo_restart_time = -1.0
        self.trash_debug_history = []
        self.trash_debug_history_limit = 80

        # 垃圾搜索分层：
        # 1. near sweep：背向投放区扫有效扇区，只接受 head/body 近目标或很近 ee 目标；
        # 2. far search：近处清空后，ee 顺时针扫有效远搜扇区并锁定单个远目标；
        # 3. approach：远目标粗靠近，本体相机看到近目标后用最新 head/body 帧微调。
        self.scan_base_heading = None
        self.scan_offsets = [math.radians(180.0)]
        self.scan_index = 0
        self.turn_target_yaw = None
        self.turn_stable_steps = 0
        self.face_near_prealign_prev_yaw = None
        self.face_near_prealign_accum_yaw = 0.0
        self.face_near_prealign_steps = 0
        self.scan_saved = set()
        self.near_sweep_start_yaw = None
        self.near_sweep_prev_yaw = None
        self.near_sweep_accum_yaw = 0.0
        self.near_sweep_span = math.radians(170.0)
        self.near_sweep_min_turn = self.near_sweep_span
        self.near_sweep_yaw_rate = 0.22
        self.near_body_max_distance = 1.60
        self.near_ee_max_distance = 0.90
        self.far_ee_min_distance = 0.75
        self.far_search_turn_steps = 0
        self.far_search_prev_yaw = None
        self.far_search_accum_yaw = 0.0
        self.far_search_span = math.radians(165.0)
        self.far_search_yaw_rate = -0.20
        self.drop_wall_safe_distance = map_prior.DROP_RADIUS + 0.45

        self.current_velocity_commands = torch.tensor(
            [0.0, 0.0, 0.0],
            device=self.device,
            dtype=torch.float32,
        ).view(1, 3)


    def get_action_spec(self) -> dict[str, dict[str, Any]] | None:
        return {}

    def _resolve_joint_ids(self, candidates: tuple[list[str], ...]) -> list[int]:
        last_error = None
        for names in candidates:
            try:
                ids, found_names = self.robot.find_joints(names)
            except ValueError as err:
                last_error = err
                continue
            if len(ids) == len(names):
                if candidates is self.ARM_JOINT_NAME_CANDIDATES:
                    self.arm_joint_names = list(found_names)
                return list(ids)
        raise ValueError(
            f"Cannot resolve required joints from candidates: {candidates}. Last error: {last_error}"
        )

    def _resolve_ee_body_name(self) -> str:
        last_error = None
        for name in self.EE_BODY_NAME_CANDIDATES:
            try:
                body_ids, _ = self.robot.find_bodies(name)
            except ValueError as err:
                last_error = err
                continue
            if len(body_ids) == 1:
                return name
        raise ValueError(
            f"Cannot resolve EE body from candidates: {self.EE_BODY_NAME_CANDIDATES}. Last error: {last_error}"
        )

    def _ensure_cartesian_targets(self):
        self.cartesian_ctrl.reset()

    def _compute_arm_overlay_action(self) -> torch.Tensor:
        self._ensure_cartesian_targets()

        arm_jpos_des = self.cartesian_ctrl.compute_base(
            self.ee_pos_target_b,
            self.ee_quat_target_b,
        )

        full_target = self.robot.data.joint_pos.clone()
        full_target[:, self.arm_ids] = arm_jpos_des
        full_target[:, self.gripper_ids] = self.gripper_open_pos.repeat(full_target.shape[0], 1)

        return (full_target - self.default_joint_pos) / self.ACTION_SCALE

    def _get_velocity_commands(self, proprio: torch.Tensor) -> torch.Tensor:
        """Return desired body velocity commands for the locomotion policy."""
        num_envs = proprio.shape[0]

        cmd = self.current_velocity_commands.to(dtype=proprio.dtype, device=self.device)
        if num_envs > 1:
            cmd = cmd.repeat(num_envs, 1)
        return cmd

    def _update_dead_reckoning(self, proprio: torch.Tensor):
        # 兼容旧函数名：实际预测逻辑已经迁移到 Localizer2D。
        # 后续视觉/激光校正也应该放进 Localizer2D，solution.py 只读 pose。
        self.localizer.predict_from_proprio(proprio)
        self.localizer.clamp_to_reasonable_bounds()

    def _correct_localization_from_observations(self, obs):
        # 视觉校正：用 head/video RGB-D 中的橙色矮墙点做世界系圆拟合。
        # 为了避免误检把位姿拉飞，estimate_world_drop_from_obs 内部已经做了
        # 半径、圆心距离、内点数量等门控；这里再用较小权重渐进修正。
        if self.step_count % 10 == 0:
            try:
                estimate = estimate_world_drop_from_obs(
                    obs,
                    self.localizer.pose_xy,
                    self.localizer.yaw,
                )
            except Exception:
                estimate = None
            if self.localizer.correct_with_drop_estimate(estimate, weight=0.18):
                self.last_visual_correction = {
                    "source": estimate.source,
                    "center_xy": estimate.center_xy.tolist(),
                    "radius": float(estimate.radius),
                    "inlier_count": int(estimate.inlier_count),
                }

        # 雷达校正：extero 是 height_scan，不是原始 xyz 点云。
        # 这里用已验证的 IsaacLab 公式 sensor_z - hit_z - 0.5 和固定 LiDAR
        # ray 方向先验，保守还原伪 2D 点；只有拟合到接近先验投放区的圆环时
        # 才小权重修正自身位置。
        if self.step_count % 15 == 0:
            try:
                lidar_estimate, lidar_diag = estimate_drop_from_height_scan(
                    obs.get("extero"),
                    self.localizer.pose_xy,
                    self.localizer.yaw,
                )
            except Exception as err:
                lidar_estimate, lidar_diag = None, {"available": False, "reason": repr(err)}
            self.last_lidar_diag = lidar_diag
            self._write_lidar_debug_snapshot(obs.get("extero"), lidar_diag)
            if self.localizer.correct_with_drop_estimate(lidar_estimate, weight=0.12):
                self.last_lidar_diag = dict(lidar_diag)
                self.last_lidar_diag["correction_used"] = True

    def _write_lidar_debug_snapshot(self, extero, lidar_diag):
        # 保存“单帧 height_scan 反推伪点云”的调试数据。
        # 这不是多帧地图，也不是仿真真值点云；它使用当前 Localizer2D 位姿
        # 把本帧 height_scan 近似投到世界系，方便画出墙体候选点和拟合圆。
        try:
            debug = debug_height_scan_projection(
                extero,
                self.localizer.pose_xy,
                self.localizer.yaw,
            )
            os.makedirs(self.lidar_debug_dir, exist_ok=True)
            tmp_path = self.lidar_debug_path + ".tmp.npz"
            np.savez_compressed(
                tmp_path,
                local_points=debug["local_points"],
                world_points=debug["world_points"],
                wall_points=debug["wall_points"],
                fit_center=debug["fit_center"],
                fit_radius=np.asarray([debug["fit_radius"]], dtype=np.float64),
                prior_center=map_prior.DROP_CENTER_XY,
                prior_radius=np.asarray([map_prior.DROP_RADIUS], dtype=np.float64),
                pose_xy=self.localizer.pose_xy,
                yaw=np.asarray([self.localizer.yaw], dtype=np.float64),
                diag_json=json.dumps(debug.get("diag", lidar_diag), ensure_ascii=False),
            )
            os.replace(tmp_path, self.lidar_debug_path)
            self.last_lidar_diag = dict(debug.get("diag", lidar_diag))
        except Exception as err:
            self.last_lidar_diag = dict(lidar_diag or {})
            self.last_lidar_diag["debug_write_error"] = repr(err)

    def _update_drop_zone_estimate(self, obs):
        # Run the relatively expensive RGB-D extraction only occasionally.
        if self.step_count % 20 != 0:
            return self.drop_tracker.estimate
        try:
            candidate = estimate_from_obs(obs)
        except Exception:
            candidate = None
        # RGB-D extraction currently yields camera-frame geometry because the
        # solution API does not expose camera extrinsics.  Keep it available
        # for debug, but do not let camera-frame centers overwrite world-frame
        # navigation until a world transform is wired in.
        if candidate is not None and candidate.source.endswith("_camera"):
            return self.drop_tracker.estimate
        return self.drop_tracker.update(candidate)

    def _compute_navigation_command(self, obs, proprio: torch.Tensor) -> torch.Tensor:
        self._update_dead_reckoning(proprio)
        self._correct_localization_from_observations(obs)
        self._update_drop_zone_estimate(obs)
        pose_xy = self.localizer.pose_xy
        yaw = self.localizer.yaw
        stand_pos_xy, _stand_heading = self.localizer.stand_pose_for_drop(stand_distance=1.5)

        if self.phase == "GO_TO_DROP_STAND":
            delta = stand_pos_xy - pose_xy
            distance = float(np.linalg.norm(delta))
            target_heading = math.atan2(float(delta[1]), float(delta[0])) if distance > 1e-6 else yaw
            heading_error = self._wrap_angle(target_heading - yaw)

            if distance < 0.45:
                self.phase = "FACE_DROP_CENTER"
                return self._velocity_tensor(0.0, 0.0, 0.0)

            # Keep the command simple for the pretrained locomotion policy:
            # rotate if badly misaligned, otherwise walk forward toward the
            # stand-off pose.  B2 starts roughly facing +X, which matches TaskB.
            if abs(heading_error) > 0.65:
                lin_x = 0.0
            else:
                lin_x = min(3.0, max(0.45, distance))
            yaw_rate = float(np.clip(1.8 * heading_error, -1.0, 1.0))
            return self._velocity_tensor(lin_x, 0.0, yaw_rate)

        if self.phase == "FACE_DROP_CENTER":
            center = map_prior.DROP_CENTER_XY
            target_heading = math.atan2(float(center[1] - pose_xy[1]), float(center[0] - pose_xy[0]))
            heading_error = self._wrap_angle(target_heading - yaw)
            if abs(heading_error) < 0.10:
                # 已经面向投放区：下一步原地慢扫附近垃圾。
                self.scan_base_heading = target_heading
                self.scan_index = 0
                self.scan_saved = set()
                # 垃圾不在投放区矮墙方向；先转到背向投放区扇区的起点，
                # 再扫约 170 度，跳过前后 90 度低信息边界画面。
                self.turn_target_yaw = self._wrap_angle(
                    target_heading + math.pi - 0.5 * self.near_sweep_span
                )
                self.turn_stable_steps = 0
                self.face_near_prealign_prev_yaw = self.localizer.yaw
                self.face_near_prealign_accum_yaw = 0.0
                self.face_near_prealign_steps = 0
                self._ensure_yolo_runner_started()
                self.phase = "FACE_NEAR_SWEEP_START"
                return self._velocity_tensor(0.0, 0.0, 0.0)
            yaw_rate = float(np.clip(1.5 * heading_error, -0.65, 0.65))
            return self._velocity_tensor(0.0, 0.0, yaw_rate)

        if self.phase == "FACE_NEAR_SWEEP_START":
            clearance_cmd = self._drop_wall_clearance_command()
            if clearance_cmd is not None:
                return clearance_cmd
            # 预对齐阶段也保存/刷新 live 帧，避免“还没开始扫图”时看起来像卡住。
            # 如果本体相机已经看到近目标，直接接管，不必等转到扇区起点。
            self._ensure_yolo_runner_started()
            self._save_live_servo_frame(obs)
            self._refresh_yolo_targets()
            near_target = self._select_near_sweep_target(self.trash_targets)
            if near_target is not None:
                self.current_trash_target = near_target
                self.ready_to_grasp_steps = 0
                self.phase = "APPROACH_TRASH_TARGET"
                return self._velocity_tensor(0.0, 0.0, 0.0)

            if self.turn_target_yaw is None:
                self.turn_target_yaw = self._wrap_angle(self.localizer.yaw + math.pi)
            heading_error = self._wrap_angle(self.turn_target_yaw - self.localizer.yaw)
            if self.face_near_prealign_prev_yaw is None:
                self.face_near_prealign_prev_yaw = self.localizer.yaw
            delta_yaw = abs(self._wrap_angle(self.localizer.yaw - self.face_near_prealign_prev_yaw))
            self.face_near_prealign_accum_yaw += delta_yaw
            self.face_near_prealign_prev_yaw = self.localizer.yaw
            self.face_near_prealign_steps += 1

            prealign_close_enough = abs(heading_error) < 0.30
            prealign_turned_enough = self.face_near_prealign_accum_yaw >= math.radians(105.0)
            prealign_timed_out = self.face_near_prealign_steps >= 90
            if prealign_close_enough or prealign_turned_enough or prealign_timed_out:
                self.near_sweep_start_yaw = self.localizer.yaw
                self.near_sweep_prev_yaw = self.localizer.yaw
                self.near_sweep_accum_yaw = 0.0
                self.turn_stable_steps = 0
                self.phase = "NEAR_SWEEP_TRASH"
                return self._velocity_tensor(0.0, 0.0, self.near_sweep_yaw_rate)
            self.turn_stable_steps = 0
            yaw_rate = float(np.clip(1.2 * heading_error, -0.50, 0.50))
            if abs(yaw_rate) < 0.32:
                yaw_rate = math.copysign(0.32, heading_error)
            return self._velocity_tensor(0.0, 0.0, yaw_rate)

        if self.phase == "NEAR_SWEEP_TRASH":
            clearance_cmd = self._drop_wall_clearance_command()
            if clearance_cmd is not None:
                return clearance_cmd
            # 慢速扫有效扇区，同时保存 head/ee live 帧给 YOLO runner。
            # 这一阶段只接受近距离目标，避免远处垃圾在附近清理前抢控制。
            self._ensure_yolo_runner_started()
            self._save_live_servo_frame(obs)
            self._refresh_yolo_targets()
            near_target = self._select_near_sweep_target(self.trash_targets)
            if near_target is not None:
                self.current_trash_target = near_target
                self.ready_to_grasp_steps = 0
                self.phase = "APPROACH_TRASH_TARGET"
                return self._velocity_tensor(0.0, 0.0, 0.0)

            if self.near_sweep_prev_yaw is None:
                self.near_sweep_prev_yaw = self.localizer.yaw
            delta_yaw = abs(self._wrap_angle(self.localizer.yaw - self.near_sweep_prev_yaw))
            self.near_sweep_accum_yaw += delta_yaw
            self.near_sweep_prev_yaw = self.localizer.yaw
            if self.near_sweep_accum_yaw >= self.near_sweep_min_turn:
                self.locked_trash_target = None
                self.locked_trash_miss_steps = 0
                self.far_locked_target = None
                self.far_locked_miss_steps = 0
                self.far_search_turn_steps = 0
                self.far_search_prev_yaw = self.localizer.yaw
                self.far_search_accum_yaw = 0.0
                self.current_trash_servo = None
                self.phase = "FAR_SEARCH_TRASH"
                return self._velocity_tensor(0.0, 0.0, 0.0)
            return self._velocity_tensor(0.0, 0.0, self.near_sweep_yaw_rate)

        if self.phase == "FAR_SEARCH_TRASH":
            clearance_cmd = self._drop_wall_clearance_command()
            if clearance_cmd is not None:
                return clearance_cmd
            self.current_trash_servo = None
            # 近扫没有近目标后，远搜改为顺时针扫一个较短有效扇区。
            # 不沿着近扫方向继续看边界/投放区矮墙；一旦 ee 看到远目标就锁定粗靠近。
            self._ensure_yolo_runner_started()
            self._save_live_servo_frame(obs)
            self._refresh_yolo_targets()
            far_target = self._select_far_search_target(self.trash_targets)
            if far_target is not None:
                self.current_trash_target = far_target
                self.ready_to_grasp_steps = 0
                self.phase = "APPROACH_TRASH_TARGET"
                return self._velocity_tensor(0.0, 0.0, 0.0)
            if self.far_search_prev_yaw is None:
                self.far_search_prev_yaw = self.localizer.yaw
            delta_yaw = abs(self._wrap_angle(self.localizer.yaw - self.far_search_prev_yaw))
            self.far_search_accum_yaw += delta_yaw
            self.far_search_prev_yaw = self.localizer.yaw
            self.far_search_turn_steps += 1
            if self.far_search_accum_yaw >= self.far_search_span:
                self.far_search_turn_steps = 0
                self.far_search_prev_yaw = self.localizer.yaw
                self.far_search_accum_yaw = 0.0
                self.phase = "NEAR_SWEEP_TRASH"
                self.near_sweep_prev_yaw = self.localizer.yaw
                self.near_sweep_accum_yaw = 0.0
                return self._velocity_tensor(0.0, 0.0, 0.0)
            return self._velocity_tensor(0.0, 0.0, self.far_search_yaw_rate)

        if self.phase == "TURN_AROUND_SCAN_READY":
            # DEPRECATED：旧版固定拍照扫描流程。当前主状态机不会进入该阶段，
            # 保留只是为了回看旧实验；不要把它接回实时搜索主路径。
            # 原地转到 180 度方向，并在稳定停住后保存图片。
            # 这个角度相对“面向投放区”的方向：180 度表示正背对投放区。
            # 控制要点：
            # 1. 每个目标 yaw 都由 scan_base_heading + 固定偏移得到，不随里程计漂移改变。
            # 2. lin_x/lin_y 始终给 0，减少拍摄时的位置变化。
            # 3. 接近目标时降低 yaw_rate，稳定若干帧后再保存，避免照片模糊和角度过冲。
            if self.turn_target_yaw is None:
                self.turn_target_yaw = self._scan_target_yaw()

            heading_error = self._wrap_angle(self.turn_target_yaw - self.localizer.yaw)
            abs_error = abs(heading_error)
            if abs_error < 0.06:
                self.turn_stable_steps += 1
                if self.turn_stable_steps >= 10:
                    self._save_scan_frame(obs)
                    self.scan_index += 1
                    self.turn_stable_steps = 0
                    if self.scan_index >= len(self.scan_offsets):
                        self._ensure_yolo_runner_started()
                        self.phase = "READY_SCAN_TRASH"
                        self.turn_target_yaw = None
                    else:
                        self.turn_target_yaw = self._scan_target_yaw()
                return self._velocity_tensor(0.0, 0.0, 0.0)

            self.turn_stable_steps = 0
            if abs_error > 0.7:
                yaw_limit = 0.55
                yaw_gain = 1.0
            elif abs_error > 0.25:
                yaw_limit = 0.35
                yaw_gain = 0.9
            else:
                yaw_limit = 0.18
                yaw_gain = 0.75
            yaw_rate = float(np.clip(yaw_gain * heading_error, -yaw_limit, yaw_limit))
            return self._velocity_tensor(0.0, 0.0, yaw_rate)

        if self.phase == "READY_SCAN_TRASH":
            # DEPRECATED：旧版 scan JSON 兜底流程。实时主路径只使用 live 帧。
            self._ensure_yolo_runner_started()
            # 兼容旧的固定拍照扫描流程；当前主流程优先走 NEAR_SWEEP_TRASH。
            # 可用 demo/tool/taskb_yolo_runner.py 检测扫描图；进入接近阶段后，
            # solution.py 会继续保存 live 当前帧，runner --watch 会持续更新同一个 JSON。
            # 约定检测结果写到下面任一 JSON：
            #   outputs/taskb_scan/<本次扫描目录>/yolo_results.json
            #   outputs/taskb_scan/<本次扫描目录>/detections.json
            #   outputs/taskb_yolo_results.json
            #
            # 这里每隔几帧轮询一次 JSON；没有结果时机器人保持静止，避免在等待识别时
            # 把已经维护好的投放区定位破坏掉。
            self._refresh_yolo_targets()
            if self.trash_targets:
                self.current_trash_target = self._select_servo_target(self.trash_targets)
                self.ready_to_grasp_steps = 0
                self.phase = "APPROACH_TRASH_TARGET"
            return self._velocity_tensor(0.0, 0.0, 0.0)

        if self.phase == "APPROACH_TRASH_TARGET":
            self._ensure_yolo_runner_started()
            previous_target = self.current_trash_target
            # 闭环视觉伺服阶段：
            # 1. 接近过程中持续保存当前 head/video 和 ee RGB-D 帧，供 YOLO runner --watch 检测；
            # 2. yolo_targets.py 会优先选择 live head/video/body 中最近的目标做精校；
            # 3. 如果本体相机当前看不到垃圾，再使用 live/scan ee 的深度和坐标粗靠近；
            # 4. 一旦本体相机重新识别到目标，排序会自动切回本体相机微调姿态。
            self._save_live_servo_frame(obs)
            before_refresh_empty_step = self.last_yolo_empty_step
            self._refresh_yolo_targets()
            refreshed_empty = self.last_yolo_empty_step != before_refresh_empty_step
            if self.trash_targets:
                self.current_trash_target = self._select_approach_target(self.trash_targets)
            if self.current_trash_target is None:
                # 近距离 head 目标在转动/靠近时可能短暂掉框。只有在 YOLO 没有明确
                # 写出空结果、且旧 live 目标仍足够新时，才短暂停住等重捕获。
                self.approach_target_lost_steps += 1
                previous_lag = self._target_image_lag_steps(previous_target)
                can_hold_previous = (
                    previous_target is not None
                    and not refreshed_empty
                    and self.approach_target_lost_steps <= 15
                    and (previous_lag is None or previous_lag <= self.yolo_max_live_lag_steps)
                )
                if can_hold_previous:
                    self.current_trash_target = previous_target
                    self.ready_to_grasp_steps = 0
                    self.ready_to_grasp_last_image = None
                    self.ready_to_grasp_last_target = None
                    return self._velocity_tensor(0.0, 0.0, 0.0)
                self.ready_to_grasp_steps = 0
                self.ready_to_grasp_last_image = None
                self.ready_to_grasp_last_target = None
                self.approach_target_lost_steps = 0
                self.current_trash_servo = None
                self.current_trash_target = None
                self.phase = "FAR_SEARCH_TRASH"
                return self._velocity_tensor(0.0, 0.0, 0.0)

            self.approach_target_lost_steps = 0
            servo = self._compute_trash_servo_command(self.current_trash_target)
            self.current_trash_servo = dict(servo)
            self._record_trash_debug_event(self.current_trash_target, servo)
            if servo.get("ready_to_grasp"):
                image_key = self._target_image_key(self.current_trash_target)
                target_key = self._ready_target_key(self.current_trash_target)
                same_target = self._ready_target_matches(self.ready_to_grasp_last_target, target_key)
                if same_target:
                    self.ready_to_grasp_steps += 1
                else:
                    self.ready_to_grasp_steps = 1
                self.ready_to_grasp_last_image = image_key
                self.ready_to_grasp_last_target = target_key
                if self.ready_to_grasp_steps >= 3:
                    # 本体相机连续多帧确认同一目标已经位于可抓取区域：
                    # 先进入趴下过渡，让机身/相机/末端整体降低，再交给机械臂抓取。
                    # 这里不直接切 READY_TO_GRASP，是为了避免腿部动作瞬间跳到低姿态。
                    self.phase = "PRONE_TRANSITION"
                    self.prone_transition_start_step = self.step_count
                return self._velocity_tensor(0.0, 0.0, 0.0)

            self.ready_to_grasp_steps = 0
            self.ready_to_grasp_last_image = None
            self.ready_to_grasp_last_target = None
            return self._velocity_tensor(
                float(servo["lin_x"]),
                float(servo["lin_y"]),
                float(servo["yaw_rate"]),
            )

        if self.phase == "PRONE_TRANSITION":
            # 本体相机近距离微调完成后，底盘停住。
            # 这里只更新状态和速度命令；真正的腿部插值在 _apply_prone_override() 完成，
            # 这样 locomotion policy、动作尺度映射和固定趴姿覆盖仍集中在动作层。
            if self.prone_transition_start_step is None:
                self.prone_transition_start_step = self.step_count
            if self.step_count - self.prone_transition_start_step >= self.prone_transition_total_steps:
                self.phase = "READY_TO_GRASP"
            return self._velocity_tensor(0.0, 0.0, 0.0)

        if self.phase == "READY_TO_GRASP":
            # 到达抓取预备位：底盘必须静止，后续机械臂抓取逻辑从这里接。
            return self._velocity_tensor(0.0, 0.0, 0.0)

        return self._velocity_tensor(0.0, 0.0, 0.0)


    def _drop_wall_clearance_command(self):
        # 搜索阶段离投放区矮墙太近时，原地转会因为足端/底盘漂移蹭墙。
        # 先退到圆环外侧安全距离，再继续近扫或远搜。
        center = map_prior.DROP_CENTER_XY
        delta = self.localizer.pose_xy - center
        distance = float(np.linalg.norm(delta))
        if distance >= self.drop_wall_safe_distance or distance < 1e-6:
            return None
        target_heading = math.atan2(float(delta[1]), float(delta[0]))
        heading_error = self._wrap_angle(target_heading - self.localizer.yaw)
        if abs(heading_error) > 0.45:
            lin_x = 0.0
        else:
            lin_x = 0.35
        yaw_rate = float(np.clip(1.2 * heading_error, -0.35, 0.35))
        return self._velocity_tensor(lin_x, 0.0, yaw_rate)


    def _ensure_yolo_runner_started(self):
        # YOLO 优先进程内后台线程，若主 Isaac 环境没装 ultralytics，
        # 自动退回到已安装的 atec-yolo conda 环境运行同一个 runner 脚本。
        if not self.yolo_enabled:
            return
        if self.yolo_started:
            if self.yolo_process is not None:
                exit_code = self.yolo_process.poll()
                if exit_code is None:
                    return
                self.last_yolo_diag = {
                    "available": False,
                    "reason": "yolo_subprocess_exited",
                    "exit_code": int(exit_code),
                }
                self._mark_yolo_runner_stopped()
            elif self.yolo_thread is not None:
                if self.yolo_thread.is_alive():
                    return
                self.last_yolo_diag = {"available": False, "reason": "embedded_yolo_thread_exited"}
                self._mark_yolo_runner_stopped()
            else:
                return
        if not self._yolo_restart_allowed():
            return
        if not os.path.exists(self.yolo_model_path):
            self.yolo_start_error = f"missing_model: {self.yolo_model_path}"
            self.last_yolo_diag = {"available": False, "reason": self.yolo_start_error}
            self.last_yolo_restart_time = time.time()
            return

        self.yolo_stop_event = threading.Event()
        try:
            runner = TaskBYoloRunner(model_path=self.yolo_model_path, conf=self.yolo_conf)
        except Exception as err:
            self.yolo_start_error = repr(err)
            if self._start_yolo_subprocess():
                self.yolo_started = True
            else:
                self.last_yolo_diag = {
                    "available": False,
                    "reason": "embedded_yolo_start_failed",
                    "error": self.yolo_start_error,
                }
                self.last_yolo_restart_time = time.time()
            return

        self.yolo_thread = threading.Thread(
            target=self._run_embedded_yolo_loop,
            args=(runner, self.yolo_stop_event),
            name="TaskBEmbeddedYolo",
            daemon=True,
        )
        self.yolo_thread.start()
        self.yolo_started = True
        self.last_yolo_restart_time = time.time()
        self.yolo_stale_count = 0
        self.last_yolo_diag = {
            "available": False,
            "reason": "embedded_yolo_started",
            "model": self.yolo_model_path,
        }

    def _run_embedded_yolo_loop(self, runner, stop_event):
        output_path = os.path.join(self.scan_dir, "yolo_results.json")
        while not stop_event.is_set():
            try:
                if os.path.isdir(self.scan_dir):
                    runner.run_scan_dir(self.scan_dir, output=output_path, max_live_per_camera=self.yolo_max_live_per_camera)
            except Exception as err:
                self.last_yolo_diag = {
                    "available": False,
                    "reason": "embedded_yolo_loop_error",
                    "error": repr(err),
                }
            stop_event.wait(max(self.yolo_watch_interval, 0.05))

    def _start_yolo_subprocess(self) -> bool:
        runner_script = os.path.join(self.project_root, "demo", "tool", "taskb_yolo_runner.py")
        if not os.path.exists(runner_script):
            return False
        try:
            self._stop_stale_yolo_processes()
            os.makedirs(os.path.dirname(self.pose_state_path), exist_ok=True)
            log_path = os.path.join(self.project_root, "outputs", "taskb_yolo_runner.log")
            self.yolo_log_handle = open(log_path, "a", encoding="utf-8")
            cmd = [
                "conda",
                "run",
                "-n",
                "atec-yolo",
                "python",
                runner_script,
                "--scan-dir",
                self.scan_dir,
                "--model",
                self.yolo_model_path,
                "--watch",
                "--interval",
                str(self.yolo_watch_interval),
                "--max-live-per-camera",
                str(self.yolo_max_live_per_camera),
            ]
            self.yolo_process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                stdout=self.yolo_log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self.last_yolo_restart_time = time.time()
            self.yolo_stale_count = 0
            self.last_yolo_diag = {
                "available": False,
                "reason": "yolo_subprocess_started",
                "pid": self.yolo_process.pid,
                "log": log_path,
            }
            return True
        except Exception as err:
            self.yolo_start_error = f"{self.yolo_start_error}; subprocess: {err!r}"
            return False

    def _stop_stale_yolo_processes(self):
        try:
            current_pid = None if self.yolo_process is None else self.yolo_process.pid
            marker = os.path.join(self.project_root, "demo", "tool", "taskb_yolo_runner.py")
            output = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
            for line in output.splitlines():
                parts = line.strip().split(None, 1)
                if len(parts) != 2:
                    continue
                pid = int(parts[0])
                args = parts[1]
                if pid == os.getpid() or pid == current_pid:
                    continue
                if marker in args:
                    try:
                        os.kill(pid, 15)
                    except Exception:
                        pass
        except Exception:
            pass

    def _stop_yolo_runner(self):
        if self.yolo_stop_event is not None:
            self.yolo_stop_event.set()
        if self.yolo_process is not None and self.yolo_process.poll() is None:
            try:
                self.yolo_process.terminate()
            except Exception:
                pass
        if self.yolo_log_handle is not None:
            try:
                self.yolo_log_handle.close()
            except Exception:
                pass
        self.yolo_log_handle = None

    def _mark_yolo_runner_stopped(self):
        self._stop_yolo_runner()
        self.yolo_started = False
        self.yolo_process = None
        self.yolo_stop_event = None
        self.yolo_thread = None

    def _yolo_restart_allowed(self) -> bool:
        last = float(getattr(self, "last_yolo_restart_time", -1.0))
        return last < 0.0 or (time.time() - last) >= self.yolo_restart_backoff_s

    def _request_yolo_restart(self, reason: str):
        self.last_yolo_diag = {**getattr(self, "last_yolo_diag", {}), "restart_reason": reason}
        if not self._yolo_restart_allowed():
            return False
        self._mark_yolo_runner_stopped()
        self.last_yolo_restart_time = time.time()
        return True

    def __del__(self):
        try:
            self._stop_yolo_runner()
        except Exception:
            pass


    def _compute_trash_servo_command(self, target):
        # live 帧是当前相机看到的目标，直接用图像误差闭环伺服。
        # DEPRECATED scan 兼容：当前主流程不再主动选择历史 scan 目标。
        # scan 只用于旧实验回放/兼容，避免机器人移动后旧 world_xy 误导控制。
        if getattr(target, "frame_kind", "scan") == "scan" and target.world_xy is not None:
            return self._world_target_servo_command(target)
        return servo_command_from_target(target)

    def _world_target_servo_command(self, target):
        # DEPRECATED：历史 scan 目标的 world_xy 粗导航，不用于当前实时主路径。
        target_xy = np.asarray(target.world_xy, dtype=np.float64)
        delta_w = target_xy - self.localizer.pose_xy
        distance = float(np.linalg.norm(delta_w))
        target_heading = math.atan2(float(delta_w[1]), float(delta_w[0])) if distance > 1e-6 else self.localizer.yaw
        heading_error = self._wrap_angle(target_heading - self.localizer.yaw)

        # 先把身体转向扫描图估计出的世界坐标，再小步前进。
        # 接近后如果本体相机看到 live 目标，会自动切回 head/video 精调。
        if abs(heading_error) > 0.45:
            lin_x = 0.0
        else:
            lin_x = float(np.clip(0.45 * (distance - 0.7), 0.0, 0.55))
        yaw_rate = float(np.clip(1.0 * heading_error, -0.45, 0.45))
        ready = False
        if distance < 0.65:
            lin_x = 0.0
            ready = False
        return {
            "lin_x": lin_x,
            "lin_y": 0.0,
            "yaw_rate": yaw_rate,
            "ready_to_grasp": ready,
            "mode": "scan_world_coarse",
            "target_distance_m": distance,
        }


    def _refresh_yolo_targets(self):
        # YOLO 结果轮询函数。
        # 检测程序只需要持续写 JSON，不需要直接改 solution.py。
        # 示例：
        # {
        #   "detections": [
        #     {
        #       "camera": "head",
        #       "bbox": [260, 210, 340, 330],
        #       "confidence": 0.91,
        #       "label": "trash",
        #       "depth_m": 0.8,
        #       "scan_angle_deg": 180
        #     },
        #     {
        #       "camera": "ee",
        #       "bbox": [400, 180, 455, 245],
        #       "confidence": 0.86,
        #       "depth_m": 2.4,
        #       "scan_angle_deg": 240
        #     }
        #   ]
        # }
        #
        # 排序策略在 demo/tool/yolo_targets.py：
        # head/video 近距离目标优先，ee 远距离目标作为粗引导；同类目标按距离近、
        # 置信度高排序。这里保持轻量，避免把检测模块和控制状态机耦合死。
        if self.step_count % 3 != 0:
            return
        targets, diag = load_yolo_targets(
            self.project_root,
            scan_dir=self.scan_dir,
            pose_xy=self.localizer.pose_xy,
            yaw=self.localizer.yaw,
            arm_qpos=self._current_arm_qpos(),
            scan_base_yaw=self.scan_base_heading,
        )
        self.last_yolo_diag = diag
        if self._yolo_result_is_stale(diag, targets):
            self.yolo_stale_count += 1
            self._clear_yolo_targets(reason="stale_yolo_result")
            if self.yolo_stale_count >= self.yolo_stale_restart_threshold:
                self._request_yolo_restart("stale_yolo_result")
                self.yolo_stale_count = 0
            return
        self.yolo_stale_count = 0
        if targets:
            self.trash_targets = targets
            if self.phase not in ("NEAR_SWEEP_TRASH", "FAR_SEARCH_TRASH", "APPROACH_TRASH_TARGET"):
                self.current_trash_target = self._select_servo_target(targets)
        elif diag.get("available"):
            # YOLO runner 写 JSON 有短暂空窗。远端 ee 目标已经锁住时，保留几帧锁定，
            # 避免显示/控制在“有框-无框”之间闪烁；超过窗口再清空，防止盲走太久。
            if self.trash_servo_mode == "ee_far_search" and self.far_locked_target is not None:
                self.far_locked_miss_steps += 1
                if self.far_locked_miss_steps <= 8:
                    self.current_trash_target = self.far_locked_target
                    return
                self.far_locked_target = None
                self.far_locked_miss_steps = 0
            self.last_yolo_empty_step = self.step_count
            self._clear_yolo_targets(reason="empty_yolo_result")

    def _clear_yolo_targets(self, reason: str):
        self.trash_targets = []
        self.current_trash_target = None
        self.locked_trash_target = None
        self.locked_trash_miss_steps = 0
        self.ready_to_grasp_steps = 0
        self.ready_to_grasp_last_image = None
        self.ready_to_grasp_last_target = None
        self.current_trash_servo = None
        self.last_yolo_diag = {**getattr(self, "last_yolo_diag", {}), "control_action": reason}

    def _yolo_result_is_stale(self, diag, targets) -> bool:
        if not diag.get("available"):
            return False
        age_s = diag.get("age_s")
        if age_s is None:
            return False
        if float(age_s) > self.yolo_max_result_age_s:
            return True
        live_steps = [self._target_live_step(t) for t in targets if getattr(t, "frame_kind", None) == "live"]
        if live_steps and self.step_count - max(live_steps) > self.yolo_max_live_lag_steps:
            return True
        return False


    def _select_approach_target(self, targets):
        # 从 ee 远搜进入接近后，继续锁定同一个远目标，避免多个远处垃圾之间来回跳。
        # 只有本体相机看到近距离目标时才允许接管，符合“远端粗靠近 -> 本体精调”。
        body_targets = [t for t in targets if self._is_valid_body_target(t)]
        if body_targets and self._body_target_can_takeover(body_targets):
            self.far_locked_target = None
            self.far_locked_miss_steps = 0
            self.trash_servo_mode = "body_track"
            self.body_target_lost_steps = 0
            return self._choose_latest_body_target(body_targets)

        if self.trash_servo_mode == "ee_far_search":
            ee_far = self._latest_targets_by_camera([
                t for t in targets
                if self._is_valid_ee_far_search_target(t)
                and t.distance_hint_m >= self.far_ee_min_distance
                and abs(t.image_error[1]) <= 0.90
            ])
            far = self._lock_or_choose_far_target(ee_far)
            if far is not None:
                return far

        return self._select_servo_target(targets)

    def _select_servo_target(self, targets):
        # 明确执行相机切换规则：
        # 1. 本体 live 相机中有有效垃圾时，进入/保持 BODY_TRACK，并只用本体目标微调；
        # 2. 本体连续丢失数轮后，才退回 live 末端相机 EE_SEARCH；
        # 3. 主实时流程不再用历史 scan 图兜底，避免机器人移动后旧世界坐标误导控制。
        body_targets = [t for t in targets if self._is_valid_body_target(t)]
        ee_targets = [t for t in targets if self._is_valid_ee_target(t)]
        scan_targets = []

        if body_targets and (self.trash_servo_mode != "scan_coarse" or self._body_target_can_takeover(body_targets)):
            self.trash_servo_mode = "body_track"
            self.body_target_lost_steps = 0
            return self._choose_latest_body_target(body_targets)

        if self.trash_servo_mode == "body_track":
            self.body_target_lost_steps += 1
            if self.body_target_lost_steps <= 4 and self.locked_trash_target is not None:
                return self.locked_trash_target
            self.locked_trash_target = None
            self.locked_trash_miss_steps = 0

        if ee_targets:
            self.trash_servo_mode = "ee_search"
            self.ee_target_lost_steps = 0
            return self._lock_or_choose(ee_targets, allow_switch_distance=0.50)

        if self.trash_servo_mode == "ee_search":
            self.ee_target_lost_steps += 1
            if self.ee_target_lost_steps <= 3 and self.locked_trash_target is not None:
                return self.locked_trash_target

        self.trash_servo_mode = "live_search"
        self.locked_trash_target = None
        self.locked_trash_miss_steps = 0
        return None

    def _choose_latest_body_target(self, targets):
        if not targets:
            return None
        latest_step = max(self._target_live_step(t) for t in targets)
        latest = [t for t in targets if self._target_live_step(t) == latest_step]
        locked_match = self._match_locked_target(latest)
        if locked_match is not None:
            chosen = locked_match
        else:
            chosen = min(latest, key=lambda t: (t.distance_hint_m, abs(t.image_error[0]), -t.confidence))
        self.locked_trash_target = chosen
        self.locked_trash_miss_steps = 0
        return chosen

    @staticmethod
    def _target_live_step(target) -> int:
        name = target.source_image or ""
        parts = str(name).split("_")
        if len(parts) >= 2 and parts[0] == "live":
            try:
                return int(parts[1])
            except Exception:
                return -1
        return -1

    def _latest_targets_by_camera(self, targets):
        if not targets:
            return []
        latest_step_by_camera = {}
        for target in targets:
            step = self._target_live_step(target)
            latest_step_by_camera[target.camera] = max(latest_step_by_camera.get(target.camera, -1), step)
        return [
            target for target in targets
            if self._target_live_step(target) == latest_step_by_camera.get(target.camera, -1)
        ]

    def _select_near_sweep_target(self, targets):
        # 近扫阶段只接受近距离、当前 live 目标。runner 会保留多张 live 图，
        # 这里按相机只使用最新 step，避免原地转动时用旧图像误差接管控制。
        body_near = self._latest_targets_by_camera([
            t for t in targets
            if self._is_valid_body_target(t) and t.distance_hint_m <= self.near_body_max_distance
        ])
        if body_near:
            self.trash_servo_mode = "body_track"
            self.body_target_lost_steps = 0
            return self._choose_latest_body_target(body_near)

        ee_near = self._latest_targets_by_camera([
            t for t in targets
            if self._is_valid_ee_target(t) and t.distance_hint_m <= self.near_ee_max_distance
        ])
        if ee_near:
            self.trash_servo_mode = "ee_near_track"
            self.ee_target_lost_steps = 0
            return self._lock_or_choose(ee_near, allow_switch_distance=0.20)
        return None

    def _select_far_search_target(self, targets):
        # 远搜阶段优先让本体近目标随时接管；只有附近确实没有目标时，
        # 才接受 ee 的远距离 live 目标或 scan 粗目标作为导航目标。
        near = self._select_near_sweep_target(targets)
        if near is not None:
            return near

        ee_far = self._latest_targets_by_camera([
            t for t in targets
            if self._is_valid_ee_far_search_target(t)
            and t.distance_hint_m >= self.far_ee_min_distance
            and abs(t.image_error[1]) <= 0.85
        ])
        if ee_far:
            self.trash_servo_mode = "ee_far_search"
            self.ee_target_lost_steps = 0
            return self._lock_or_choose_far_target(ee_far)

        return None

    @staticmethod
    def _target_image_key(target) -> str | None:
        if target is None or getattr(target, "frame_kind", None) != "live":
            return None
        return getattr(target, "source_image", None)

    def _target_image_lag_steps(self, target) -> int | None:
        step = self._target_live_step(target) if target is not None else -1
        return None if step < 0 else int(self.step_count - step)

    @staticmethod
    def _ready_target_key(target):
        if target is None:
            return None
        x1, y1, x2, y2 = target.bbox_xyxy
        w, h = target.image_size
        return {
            "camera": target.camera,
            "label": target.label,
            "center": ((x1 + x2) / max(2.0 * w, 1.0), (y1 + y2) / max(2.0 * h, 1.0)),
            "distance": float(target.distance_hint_m),
        }

    @staticmethod
    def _ready_target_matches(previous, current) -> bool:
        if current is None:
            return False
        if previous is None:
            return True
        if previous.get("camera") != current.get("camera") or previous.get("label") != current.get("label"):
            return False
        pcx, pcy = previous.get("center", (0.0, 0.0))
        ccx, ccy = current.get("center", (0.0, 0.0))
        if float(np.hypot(pcx - ccx, pcy - ccy)) > 0.18:
            return False
        pd = float(previous.get("distance", 0.0))
        cd = float(current.get("distance", 0.0))
        return abs(pd - cd) <= 0.35

    def _record_trash_debug_event(self, target, servo):
        try:
            event = {
                "step": int(self.step_count),
                "phase": self.phase,
                "mode": self.trash_servo_mode,
                "ready_to_grasp_steps": int(self.ready_to_grasp_steps),
                "servo": None if servo is None else {
                    "ready_to_grasp": bool(servo.get("ready_to_grasp", False)),
                    "hold_for_grasp": bool(servo.get("hold_for_grasp", False)),
                    "lin_x": float(servo.get("lin_x", 0.0)),
                    "lin_y": float(servo.get("lin_y", 0.0)),
                    "yaw_rate": float(servo.get("yaw_rate", 0.0)),
                },
                "target": None if target is None else {
                    "camera": target.camera,
                    "label": target.label,
                    "source_image": target.source_image,
                    "image_error": list(target.image_error),
                    "distance_hint_m": float(target.distance_hint_m),
                    "confidence": float(target.confidence),
                    "bbox_xyxy": list(target.bbox_xyxy),
                },
            }
            self.trash_debug_history.append(event)
            if len(self.trash_debug_history) > self.trash_debug_history_limit:
                self.trash_debug_history = self.trash_debug_history[-self.trash_debug_history_limit:]
        except Exception:
            pass

    def _lock_or_choose_far_target(self, candidates):
        if not candidates:
            self.far_locked_miss_steps += 1
            if self.far_locked_miss_steps > 8:
                self.far_locked_target = None
            return self.far_locked_target if self.far_locked_miss_steps <= 8 else None

        # 第一次远搜锁定：优先选最新 ee 帧里更近的目标，再看居中和置信度。
        # 锁定后只按相似度跟踪，不因为另一个目标稍近就频繁切换。
        if self.far_locked_target is None:
            self.far_locked_target = min(
                candidates,
                key=lambda t: (t.distance_hint_m, abs(t.image_error[0]) + 0.35 * abs(t.image_error[1]), -t.confidence),
            )
            self.far_locked_miss_steps = 0
            self.locked_trash_target = self.far_locked_target
            self.locked_trash_miss_steps = 0
            return self.far_locked_target

        match = self._match_far_target(self.far_locked_target, candidates)
        if match is not None:
            self.far_locked_target = match
            self.far_locked_miss_steps = 0
            self.locked_trash_target = match
            self.locked_trash_miss_steps = 0
            return match

        self.far_locked_miss_steps += 1
        if self.far_locked_miss_steps <= 8:
            return self.far_locked_target

        self.far_locked_target = min(
            candidates,
            key=lambda t: (t.distance_hint_m, abs(t.image_error[0]) + 0.35 * abs(t.image_error[1]), -t.confidence),
        )
        self.far_locked_miss_steps = 0
        self.locked_trash_target = self.far_locked_target
        self.locked_trash_miss_steps = 0
        return self.far_locked_target

    def _match_far_target(self, locked, candidates):
        if locked is None:
            return None
        if locked.world_xy is not None:
            best = None
            best_dist = float("inf")
            locked_xy = np.asarray(locked.world_xy, dtype=np.float64)
            for target in candidates:
                if target.world_xy is None:
                    continue
                dist = float(np.linalg.norm(np.asarray(target.world_xy, dtype=np.float64) - locked_xy))
                if dist < best_dist:
                    best = target
                    best_dist = dist
            if best is not None and best_dist < 1.25:
                return best
        return self._match_specific_target(locked, candidates, max_score=1.20)

    def _lock_or_choose(self, candidates, allow_switch_distance: float):
        if not candidates:
            return None
        best = min(candidates, key=lambda t: (t.distance_hint_m, -t.confidence))
        locked_match = self._match_locked_target(candidates)
        if locked_match is None:
            self.locked_trash_miss_steps += 1
            if self.locked_trash_miss_steps > 2 or self.locked_trash_target is None:
                self.locked_trash_target = best
                self.locked_trash_miss_steps = 0
            return self.locked_trash_target

        self.locked_trash_miss_steps = 0
        if best.distance_hint_m + allow_switch_distance < locked_match.distance_hint_m:
            self.locked_trash_target = best
        else:
            self.locked_trash_target = locked_match
        return self.locked_trash_target

    def _match_locked_target(self, targets):
        locked = self.locked_trash_target
        if locked is None:
            return None
        return self._match_specific_target(locked, targets, max_score=0.45)

    def _match_specific_target(self, locked, targets, max_score: float):
        best = None
        best_score = float("inf")
        for target in targets:
            if target.frame_kind != locked.frame_kind or target.camera != locked.camera:
                continue
            score = self._target_similarity_score(locked, target)
            if score < best_score:
                best = target
                best_score = score
        return best if best_score < max_score else None

    @staticmethod
    def _target_similarity_score(a, b) -> float:
        ax, ay = a.image_error
        bx, by = b.image_error
        image_delta = float(np.hypot(ax - bx, ay - by))
        depth_delta = abs(float(a.distance_hint_m) - float(b.distance_hint_m)) / 2.0
        world_delta = 0.0
        if a.world_xy is not None and b.world_xy is not None:
            world_delta = float(np.linalg.norm(np.asarray(a.world_xy) - np.asarray(b.world_xy))) / 2.0
        return image_delta + depth_delta + world_delta

    @staticmethod
    def _is_valid_body_target(target) -> bool:
        if target.frame_kind != "live" or target.camera not in ("head", "video", "body"):
            return False
        if not (0.25 <= target.distance_hint_m <= 2.2):
            return False
        if AlgSolution._target_bbox_ok(target, max_area=0.35, reject_touch=2):
            return True
        # 近距离时垃圾框会变大/贴边。只要深度可信且目标仍在画面有效区域，
        # 继续保持/微调，不要丢目标回到搜索转圈。
        return target.distance_hint_m <= 1.35 and AlgSolution._target_bbox_close_ok(target)

    @staticmethod
    def _body_target_can_takeover(targets) -> bool:
        # scan 粗导航阶段会先转向 180 度扫描图中的目标。回转/靠近途中，
        # 本体相机可能短暂扫到别的垃圾；只有距离已经进入近距离微调范围时，
        # 才允许本体相机接管，避免中途换去另一张图里的目标。
        return any(t.distance_hint_m <= 1.1 and abs(t.image_error[0]) <= 0.45 for t in targets)

    @staticmethod
    def _is_valid_ee_target(target) -> bool:
        if target.frame_kind != "live" or target.camera != "ee":
            return False
        if not AlgSolution._target_bbox_ok(target, max_area=0.45, reject_touch=2):
            return False
        # ee 相机是粗搜索用；如果反投影到狗本体系后方，当前底盘伺服不能直接前进追它。
        if target.point_body is not None and float(target.point_body[0]) < 0.10:
            return False
        return 0.20 <= target.distance_hint_m <= 6.0

    @staticmethod
    def _is_valid_ee_far_search_target(target) -> bool:
        if target.frame_kind != "live" or target.camera != "ee":
            return False
        if not AlgSolution._target_bbox_ok(target, max_area=0.45, reject_touch=2):
            return False
        # 远搜阶段 ee 只做粗引导，不把 FK 反投影的 body_x 作为硬门控。
        # 机械臂姿态/外参稍有偏差时，真实可见目标会被算到狗身后，导致一直转圈。
        return 0.20 <= target.distance_hint_m <= 6.0

    @staticmethod
    def _is_scan_target(target) -> bool:
        return target.frame_kind == "scan" and target.world_xy is not None

    @staticmethod
    def _target_bbox_close_ok(target) -> bool:
        w, h = target.image_size
        x1, y1, x2, y2 = target.bbox_xyxy
        if x2 <= x1 or y2 <= y1:
            return False
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1) / max(float(w * h), 1.0)
        if area > 0.70:
            return False
        # 允许近处框贴一个边，尤其是底边；多边大面积贴边通常是误检。
        touch = int(x1 <= 3.0) + int(y1 <= 3.0) + int(x2 >= w - 3.0) + int(y2 >= h - 3.0)
        if touch >= 3:
            return False
        err_x, err_y = target.image_error
        return abs(err_x) <= 0.75 and abs(err_y) <= 0.80

    @staticmethod
    def _target_bbox_ok(target, max_area: float, reject_touch: int) -> bool:
        w, h = target.image_size
        x1, y1, x2, y2 = target.bbox_xyxy
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1) / max(float(w * h), 1.0)
        touch = int(x1 <= 3.0) + int(y1 <= 3.0) + int(x2 >= w - 3.0) + int(y2 >= h - 3.0)
        return area <= max_area and touch < reject_touch

    @staticmethod
    def _is_live_head_target(target) -> bool:
        return target.frame_kind == "live" and target.camera in ("head", "video", "body")

    @staticmethod
    def _target_priority_key(target) -> int:
        if target.frame_kind == "live" and target.camera in ("head", "video", "body"):
            return 0
        if target.frame_kind == "live" and target.camera == "ee":
            return 1
        if target.camera in ("head", "video", "body"):
            return 2
        return 3


    def _current_arm_qpos(self):
        # 末端相机是装在机械臂上的，想把 ee 相机坐标点转回狗本体系，
        # 需要当前 Piper 六个关节角做 FK。这里从 proprio 的 joint_pos 段读取。
        # 如果读取失败，yolo_targets.py 会退回默认机械臂姿态，只作为远距离粗引导。
        try:
            proprio = self._last_proprio_for_arm
            action_dim = (int(proprio.shape[-1]) - 12) // 3
            joint_pos_all = proprio[:, 12:12 + action_dim]
            arm_q = joint_pos_all[0, self.arm_joint_indices[:6]].detach().cpu().numpy()
            return arm_q
        except Exception:
            return None


    def _write_pose_state(self):
        # 给 demo/test/pose_viewer.py 使用的轻量状态文件。比赛策略不依赖它；
        # 写失败时直接忽略，避免可视化影响控制。
        try:
            os.makedirs(os.path.dirname(self.pose_state_path), exist_ok=True)
            state = {
                "step": int(self.step_count),
                "phase": self.phase,
                "pose": {
                    "xy": self.localizer.pose_xy.tolist(),
                    "yaw": float(self.localizer.yaw),
                    "last_correction": self.localizer.last_correction,
                },
                "map_bounds": dict(map_prior.MAP_BOUNDS),
                "drop_zone": {
                    "center_xy": map_prior.DROP_CENTER_XY.tolist(),
                    "radius": float(map_prior.DROP_RADIUS),
                },
                "visual_correction": self.last_visual_correction,
                "lidar_diag": self.last_lidar_diag,
                "yolo_diag": self.last_yolo_diag,
                "control": {
                    "velocity_command": self.current_velocity_commands.detach().cpu().numpy().reshape(-1).tolist(),
                    "leg_policy": self.active_leg_policy_name,
                    "turn_target_yaw": None if self.turn_target_yaw is None else float(self.turn_target_yaw),
                    "near_sweep_accum_yaw": float(self.near_sweep_accum_yaw),
                    "far_search_accum_yaw": float(self.far_search_accum_yaw),
                    "far_search_yaw_rate": float(self.far_search_yaw_rate),
                    "face_near_prealign_accum_yaw": float(self.face_near_prealign_accum_yaw),
                    "face_near_prealign_steps": int(self.face_near_prealign_steps),
                },
                "trash": {
                    "target_count": len(self.trash_targets),
                    "current_target": (
                        None
                        if self.current_trash_target is None
                        else self.current_trash_target.to_debug_dict()
                    ),
                    "current_target_image_lag_steps": self._target_image_lag_steps(self.current_trash_target),
                    "servo": self.current_trash_servo,
                    "ready_to_grasp_steps": int(self.ready_to_grasp_steps),
                    "ready_to_grasp_last_image": self.ready_to_grasp_last_image,
                    "ready_to_grasp_last_target": self.ready_to_grasp_last_target,
                    "approach_target_lost_steps": int(self.approach_target_lost_steps),
                    "last_yolo_empty_step": int(getattr(self, "last_yolo_empty_step", -1)),
                    "far_locked_target": (
                        None
                        if self.far_locked_target is None
                        else self.far_locked_target.to_debug_dict()
                    ),
                    "far_locked_miss_steps": int(self.far_locked_miss_steps),
                    "debug_history": list(getattr(self, "trash_debug_history", [])),
                },
                "prior_errors": self.prior_errors,
            }
            tmp_path = self.pose_state_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.pose_state_path)
        except Exception:
            pass

    def _scan_target_yaw(self) -> float:
        # 当前扫描目标角。scan_base_heading 是面向投放区方向，
        # scan_offsets 当前只包含 180 度。
        if self.scan_base_heading is None:
            self.scan_base_heading = self.localizer.yaw
        offset = self.scan_offsets[min(self.scan_index, len(self.scan_offsets) - 1)]
        return self._wrap_angle(self.scan_base_heading + offset)

    def _save_scan_frame(self, obs):
        # 保存当前方向的 RGB-D 图像，供后续离线 YOLO / 深度反投影调试。
        # 文件名里带角度偏移，方便把识别结果映射回扫描方向。
        # 现在会保存两类视角：
        # 1. head/video：狗头低位相机，适合近距离精调；
        # 2. ee：机械臂末端相机，视场更宽，适合远搜粗引导和补近处盲区。
        if self.scan_index in self.scan_saved:
            return
        image = obs.get("image") if isinstance(obs, dict) else None
        if not isinstance(image, dict):
            return

        try:
            from PIL import Image
            os.makedirs(self.scan_dir, exist_ok=True)
            angle_deg = int(round(math.degrees(self.scan_offsets[self.scan_index])))

            saved_any = False
            primary_rgb = image.get("head_rgb")
            primary_depth = image.get("head_depth")
            primary_name = "head"
            if primary_rgb is None and image.get("video_rgb") is not None:
                primary_rgb = image.get("video_rgb")
                primary_depth = image.get("video_depth")
                primary_name = "video"

            saved_any |= self._save_camera_frame(
                Image,
                primary_name,
                primary_rgb,
                primary_depth,
                angle_deg,
            )
            saved_any |= self._save_camera_frame(
                Image,
                "ee",
                image.get("ee_rgb"),
                image.get("ee_depth"),
                angle_deg,
            )

            # 即使某一路缺图，也不要在同一个角度反复卡住；保存到至少一路即可。
            if saved_any:
                self.scan_saved.add(self.scan_index)
        except Exception:
            # 拍照失败不影响控制流程，后续可以在日志里加更详细的错误输出。
            self.scan_saved.add(self.scan_index)

    def _save_live_servo_frame(self, obs):
        # 接近阶段保存当前帧，配合 `python demo/tool/taskb_yolo_runner.py --watch`
        # 实现本体/末端相机的闭环检测。文件名带 live 前缀，目标排序会优先使用
        # live head/video/body 的最近目标；本体没有目标时才退回 ee 粗定位。
        if self.step_count - self.last_live_frame_step < self.live_frame_interval:
            return
        image = obs.get("image") if isinstance(obs, dict) else None
        if not isinstance(image, dict):
            return

        try:
            from PIL import Image
            os.makedirs(self.scan_dir, exist_ok=True)
            saved_any = False

            primary_rgb = image.get("head_rgb")
            primary_depth = image.get("head_depth")
            primary_name = "head"
            if primary_rgb is None and image.get("video_rgb") is not None:
                primary_rgb = image.get("video_rgb")
                primary_depth = image.get("video_depth")
                primary_name = "video"

            saved_any |= self._save_camera_frame(
                Image,
                primary_name,
                primary_rgb,
                primary_depth,
                angle_deg=None,
                prefix=f"live_{self.step_count:06d}_{primary_name}",
            )
            saved_any |= self._save_camera_frame(
                Image,
                "ee",
                image.get("ee_rgb"),
                image.get("ee_depth"),
                angle_deg=None,
                prefix=f"live_{self.step_count:06d}_ee",
            )
            if saved_any:
                self.last_live_frame_step = self.step_count
        except Exception:
            # 实时拍照失败时继续使用上一轮 YOLO 结果，避免控制循环被感知异常打断。
            self.last_live_frame_step = self.step_count

    def _save_camera_frame(self, image_cls, camera_name: str, rgb, depth, angle_deg: int | None, prefix: str | None = None) -> bool:
        if rgb is None:
            return False

        if prefix is None:
            prefix = f"scan_{self.scan_index:02d}_{angle_deg:03d}deg_{camera_name}"
        rgb_np = self._tensor_to_numpy(rgb)
        if rgb_np.ndim == 4:
            rgb_np = rgb_np[0]
        rgb_np = np.asarray(rgb_np, dtype=np.uint8)
        image_cls.fromarray(rgb_np).save(os.path.join(self.scan_dir, prefix + "_rgb.png"))

        if depth is not None:
            depth_np = self._tensor_to_numpy(depth)
            if depth_np.ndim == 4:
                depth_np = depth_np[0]
            if depth_np.ndim == 3:
                depth_np = depth_np[..., 0]
            depth_np = np.asarray(depth_np, dtype=np.float32)
            np.save(os.path.join(self.scan_dir, prefix + "_depth.npy"), depth_np)
            depth_mm = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0)
            depth_mm = np.clip(depth_mm * 1000.0, 0.0, 65535.0).astype(np.uint16)
            image_cls.fromarray(depth_mm).save(os.path.join(self.scan_dir, prefix + "_depth_mm.png"))

        return True

    @staticmethod
    def _tensor_to_numpy(value):
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            return value.numpy()
        return np.asarray(value)

    def _velocity_tensor(self, lin_x: float, lin_y: float, yaw_rate: float) -> torch.Tensor:
        return torch.tensor(
            [lin_x, lin_y, yaw_rate],
            device=self.device,
            dtype=torch.float32,
        ).view(1, 3)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def _extract_policy_obs(self, obs, action_dim) -> torch.Tensor:
        # 原始 proprio 排布：
        # base_lin_vel(3), base_ang_vel(3), env_velocity_cmd(3), projected_gravity(3),
        # joint_pos(action_dim), joint_vel(action_dim), last_actions(action_dim)。
        # 腿部 policy 只吃 12 个腿关节相关观测，机械臂动作由默认值覆盖。
        proprio = obs["proprio"].to(self.device)

        idx = 0
        _base_lin_vel = proprio[:, idx:idx + 3]
        idx += 3

        base_ang_vel = proprio[:, idx:idx + 3]
        idx += 3

        _velocity_commands_env = proprio[:, idx:idx + 3]
        idx += 3

        projected_gravity = proprio[:, idx:idx + 3]
        idx += 3

        joint_pos_all = proprio[:, idx:idx + action_dim]
        idx += action_dim

        joint_vel_all = proprio[:, idx:idx + action_dim]
        idx += action_dim

        actions_all = proprio[:, idx:idx + action_dim]

        joint_pos_leg = joint_pos_all[:, self.leg_joint_indices]
        joint_vel_leg = joint_vel_all[:, self.leg_joint_indices]
        actions_env_leg = actions_all[:, self.leg_joint_indices]

        actions_train_leg = actions_env_leg * self.env_to_train_action_scale.to(dtype=proprio.dtype)
        velocity_commands = self._get_velocity_commands(proprio)

        policy_obs = torch.cat(
            [
                base_ang_vel * 0.25,
                projected_gravity,
                velocity_commands,
                joint_pos_leg,
                joint_vel_leg * 0.05,
                actions_train_leg,
            ],
            dim=-1,
        )

        return policy_obs

    def _map_policy_action_to_env_action(self, action_train: torch.Tensor, action_dim: int) -> torch.Tensor:
        """把腿部 policy 输出映射到当前环境的全身 action。

        policy.pt 输出的是训练时的 12 维腿部动作；环境实际需要 B2Piper 的 20 维动作：
        前 12 维是腿，后 8 维是 Piper 机械臂/夹爪。腿部还要乘以
        train_to_env_action_scale，补偿训练环境和比赛环境的 action scale 差异。
        """
        if action_train.shape[-1] != self.leg_action_dim:
            raise ValueError(
                f"Policy output dim mismatch: got {action_train.shape[-1]}, expected {self.leg_action_dim}"
            )

        num_envs = action_train.shape[0]
        leg_action_env = action_train * self.train_to_env_action_scale

        action_env = torch.zeros(
            (num_envs, action_dim),
            device=self.device,
            dtype=torch.float32,
        )

        action_env[:, self.leg_joint_indices] = leg_action_env
        action_env[:, self.arm_joint_indices] = self.arm_default_action.repeat(num_envs, 1)

        return action_env

    def _prone_override_alpha(self) -> float | None:
        """返回固定趴姿覆盖权重。

        None 表示当前不是预抓取趴下相关阶段，不改腿部动作；
        0 到 1 表示正在 PRONE_TRANSITION 中线性插值；
        1 表示已经进入 READY_TO_GRASP，需要持续保持固定趴姿。
        """
        if self.phase == "READY_TO_GRASP":
            return 1.0
        if self.phase != "PRONE_TRANSITION":
            return None
        if self.prone_transition_start_step is None:
            self.prone_transition_start_step = self.step_count
        elapsed_steps = max(0, self.step_count - self.prone_transition_start_step)
        return float(np.clip(elapsed_steps / self.prone_transition_total_steps, 0.0, 1.0))

    def _apply_prone_override(self, action_env: torch.Tensor) -> torch.Tensor:
        """在预抓取阶段覆盖腿部 12 维动作，机械臂动作不变。

        这里使用“RL 当前输出 -> 固定趴姿”的动作空间插值，而不是直接插值关节角。
        因为 position action 本身就是相对默认关节角的目标偏移，插值 action 等价于
        插值目标关节角，并且能保持与 get_action_spec() 的 scale 定义一致。
        """
        alpha = self._prone_override_alpha()
        if alpha is None:
            return action_env
        prone = self.fixed_prone_leg_action.repeat(action_env.shape[0], 1)
        action_env[:, self.leg_joint_indices] = (
            (1.0 - alpha) * action_env[:, self.leg_joint_indices] + alpha * prone
        )
        return action_env

    def _select_leg_policy(self):
        # 趴下阶段仍运行 locomotion policy 作为插值起点；最终腿部动作会在
        # _apply_prone_override() 中被固定趴姿覆盖。active_leg_policy_name 只用于调试状态。
        if self.phase in ("PRONE_TRANSITION", "READY_TO_GRASP"):
            self.active_leg_policy_name = "fixed_prone"
            return self.policy
        self.active_leg_policy_name = "locomotion"
        return self.policy
    
    def predicts(self, obs, current_score):
        """每个控制步的总入口：更新策略状态、跑腿部 policy、返回完整动作。

        当前动作结构是“腿部 RL + 预抓取固定趴姿覆盖 + 机械臂默认保持”：
        1. _compute_navigation_command() 根据状态机产生期望底盘速度；
        2. _extract_policy_obs() 把速度命令塞进腿部 policy 观测；
        3. policy 输出 12 维腿部动作，作为正常行走/站立控制；
        4. _map_policy_action_to_env_action() 映射成环境 20 维全身 action；
        5. 如果处于 PRONE_TRANSITION / READY_TO_GRASP，_apply_prone_override()
           会把前 12 维腿部动作平滑覆盖为固定趴姿，后 8 维机械臂仍保持默认。

        后续接机械臂抓取时，推荐从两个位置切入：
        - 状态层：在 self.phase == "READY_TO_GRASP" 后增加抓取/夹爪/投放子状态。
        - 动作层：把 _map_policy_action_to_env_action() 里 arm_joint_indices 对应的
          self.arm_default_action 替换成你的机械臂控制输出。
        """
        # current_score 已经超过任务成功阈值时直接 giveup，避免成功后继续动作扣分。
        if current_score > 1:
            return {'action': [], 'giveup': True}

        # proprio 是本帧低维状态，也是后面腿部 policy 和末端相机 FK 都要用的数据。
        # _last_proprio_for_arm 会被 _current_arm_qpos() 读取，用于把 ee 相机检测点
        # 通过 Piper FK 转回机器人本体系。
        proprio = obs["proprio"].to(self.device)
        self._last_proprio_for_arm = proprio
        self.step_count += 1

        # 状态机核心：
        # GO_TO_DROP_STAND -> FACE_DROP_CENTER -> FACE_NEAR_SWEEP_START ->
        # NEAR_SWEEP_TRASH -> FAR_SEARCH_TRASH/APPROACH_TRASH_TARGET -> READY_TO_GRASP。
        # 这里返回的是期望底盘速度 [lin_x, lin_y, yaw_rate]，不是最终 action。
        self.current_velocity_commands = self._compute_navigation_command(obs, proprio)
        leg_policy = self._select_leg_policy()

        # 写调试状态给 demo/test/pose_viewer.py。这个 JSON 只用于可视化，
        # 不反向参与控制，所以写失败也不会影响策略。
        if self.step_count % 5 == 0:
            self._write_pose_state()

        # 从 proprio 总长度反推出当前环境 action_dim。当前 B2Piper 通常是 20：
        # 前 12 个腿关节 + 后 8 个机械臂/夹爪关节。
        action_dim = (int(proprio.shape[-1]) - 12) // 3

        # 组装腿部 policy 观测。注意 current_velocity_commands 会在这里进入 policy，
        # 所以导航状态机并不直接控制关节，而是给 RL 行走策略下速度命令。
        policy_obs = self._extract_policy_obs(obs, action_dim)

        # TorchScript 腿部策略推理，输出训练时的 12 维腿部动作空间。
        with torch.inference_mode():
            action_train = leg_policy(policy_obs)

        # 兼容 policy 返回 numpy/list 的情况，统一转成 GPU/CPU 上的 float32 Tensor。
        if not isinstance(action_train, torch.Tensor):
            action_train = torch.as_tensor(
                action_train, device=self.device, dtype=torch.float32
            )

        action_train = action_train.to(device=self.device, dtype=torch.float32)

        # 单环境时有些模型可能返回 [12]，这里补 batch 维成 [1, 12]。
        if action_train.ndim == 1:
            action_train = action_train.unsqueeze(0)

        # 映射到环境全身动作：
        # - 腿部：policy 输出按训练/环境尺度转换后写入 0:12。
        # - 机械臂：当前写入 arm_default_action。接抓取程序时，优先改这里，
        #   或者新增 _compute_arm_grasp_action() 后在 READY_TO_GRASP/抓取阶段覆盖。
        action_env = self._map_policy_action_to_env_action(action_train, action_dim)
        action_env = self._apply_prone_override(action_env)

        # 评测接口要求 Python list，不接受 Tensor。
        action_env = action_env.cpu().numpy().tolist()
        return {'action': action_env, 'giveup': False}

