import os
import torch
from typing import Any
import math
import time
import json
import numpy as np

try:
    from demo.tool.drop_zone import DropZoneTracker, estimate_from_obs, estimate_world_drop_from_obs
    from demo.tool.localizer2d import Localizer2D
    from demo.tool import taskb_map_prior as map_prior
    from demo.tool.piper_kinematics import validate_piper_kinematics
except ImportError:
    from tool.drop_zone import DropZoneTracker, estimate_from_obs, estimate_world_drop_from_obs
    from tool.localizer2d import Localizer2D
    from tool import taskb_map_prior as map_prior
    from tool.piper_kinematics import validate_piper_kinematics

class AlgSolution:

    ACTION_SCALE = 0.5
    EE_BODY_NAME_CANDIDATES = ("gripper_base", "piper_gripper_base")
    ARM_JOINT_NAME_CANDIDATES = (
        ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        ["arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5", "arm_joint6"],
    )

    def __init__(self):
        policy_path = os.path.dirname(os.path.abspath(__file__)) + '/policy.pt'
        self.device = 'cuda'

        self.policy = torch.jit.load(policy_path, map_location=self.device)
        self.policy.eval()

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

        # Fixed zero base velocity command for policy input.
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

        self.reset()

    def reset(self, **kwargs):
        self.drop_tracker = DropZoneTracker()
        self.test_mode = os.environ.get("ATEC_TEST_MODE", "").strip().lower()
        self.phase = "TEST_ZIGZAG" if self.test_mode == "zigzag" else "GO_TO_DROP_STAND"
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

        # 扫描阶段的目标角只在进入阶段时按固定角度序列生成。
        # 如果边转边根据漂移后的 pose 重新计算目标角，容易出现多转 20~30 度。
        # 这里按“面向投放区的方向”为 0 度，依次转到 120/180/240 度停下拍照。
        self.scan_base_heading = None
        self.scan_offsets = [math.radians(120.0), math.radians(180.0), math.radians(240.0)]
        self.scan_index = 0
        self.turn_target_yaw = None
        self.turn_stable_steps = 0
        self.scan_saved = set()
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.scan_dir = os.path.join(
            self.project_root,
            "outputs",
            "taskb_scan",
            time.strftime("%Y%m%d_%H%M%S"),
        )
        self.pose_state_path = os.path.join(self.project_root, "outputs", "taskb_pose_state.json")

        self.current_velocity_commands = torch.tensor(
            [0.0, 0.0, 0.0],
            device=self.device,
            dtype=torch.float32,
        ).view(1, 3)

        # 定位校验专用 Z 字形路径。只在 ATEC_TEST_MODE=zigzag 时启用。
        # 点位在修正后的 TaskB 世界地图 [-20,0]x[-20,0] 内，尽量覆盖左/右/上/下区域。
        self.test_waypoints = [
            np.array([-16.0, -16.0], dtype=np.float64),
            np.array([-4.0, -16.0], dtype=np.float64),
            np.array([-4.0, -13.0], dtype=np.float64),
            np.array([-16.0, -13.0], dtype=np.float64),
            np.array([-16.0, -10.0], dtype=np.float64),
            np.array([-4.0, -10.0], dtype=np.float64),
            np.array([-4.0, -7.0], dtype=np.float64),
            np.array([-16.0, -7.0], dtype=np.float64),
            np.array([-16.0, -4.0], dtype=np.float64),
            np.array([-4.0, -4.0], dtype=np.float64),
            np.array([-10.0, -10.0], dtype=np.float64),
        ]
        self.test_waypoint_index = 0
        self.test_loop_count = 0


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

        # 雷达校正：当前 extero 是处理后的 height_scan，不是原始 ray hit/range。
        # 先做诊断和接口占位，等确认排列和几何意义后再启用边界线匹配。
        if self.step_count % 25 == 0:
            self.last_lidar_diag = self.localizer.inspect_extero_for_lidar(obs.get("extero"))

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
        if self.test_mode == "zigzag":
            return self._compute_zigzag_test_command()

        estimate = self._update_drop_zone_estimate(obs)
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
                # 已经面向投放区：下一步开始做三方向扫描。
                # scan_base_heading 记录“面向投放区”的方向，后续目标角都用它加固定偏移，
                # 不再跟随里程计位置漂移实时变化。
                self.scan_base_heading = target_heading
                self.scan_index = 0
                self.scan_saved = set()
                self.turn_target_yaw = self._scan_target_yaw()
                self.turn_stable_steps = 0
                self.phase = "TURN_AROUND_SCAN_READY"
                return self._velocity_tensor(0.0, 0.0, 0.0)
            yaw_rate = float(np.clip(1.5 * heading_error, -0.65, 0.65))
            return self._velocity_tensor(0.0, 0.0, yaw_rate)

        if self.phase == "TURN_AROUND_SCAN_READY":
            # 原地转到 120/180/240 度三个方向，并在每个方向稳定停住后保存图片。
            # 这些角度都是相对“面向投放区”的方向：
            #   120 度：偏左后方，先看一侧垃圾区域；
            #   180 度：正背对投放区，看主要垃圾区域；
            #   240 度：偏右后方，补另一侧视野。
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

        return self._velocity_tensor(0.0, 0.0, 0.0)



    def _compute_zigzag_test_command(self) -> torch.Tensor:
        # 定位校验模式：不执行投放区靠近/扫描状态机，只沿固定 Z 字形路径跑。
        # 你可以同时打开 IsaacSim 和 demo/test/pose_viewer.py，对比仿真中的实际位置
        # 与维护出来的蓝色 pose 是否一致。测试结束后删除本函数和 run 脚本即可。
        pose_xy = self.localizer.pose_xy
        yaw = self.localizer.yaw
        if not self.test_waypoints:
            return self._velocity_tensor(0.0, 0.0, 0.0)

        target = self.test_waypoints[self.test_waypoint_index]
        delta = target - pose_xy
        distance = float(np.linalg.norm(delta))
        if distance < 0.65:
            self.test_waypoint_index += 1
            if self.test_waypoint_index >= len(self.test_waypoints):
                self.test_waypoint_index = 0
                self.test_loop_count += 1
            target = self.test_waypoints[self.test_waypoint_index]
            delta = target - pose_xy
            distance = float(np.linalg.norm(delta))

        target_heading = math.atan2(float(delta[1]), float(delta[0])) if distance > 1e-6 else yaw
        heading_error = self._wrap_angle(target_heading - yaw)

        # 误差大时先原地转，误差小时再前进，减少横向漂移对定位校验的干扰。
        if abs(heading_error) > 0.65:
            lin_x = 0.0
        else:
            lin_x = float(np.clip(distance, 0.45, 2.2))
        yaw_rate = float(np.clip(1.6 * heading_error, -0.9, 0.9))
        self.phase = f"TEST_ZIGZAG_{self.test_waypoint_index:02d}"
        return self._velocity_tensor(lin_x, 0.0, yaw_rate)

    def _write_pose_state(self):
        # 给 demo/test/pose_viewer.py 使用的轻量状态文件。比赛策略不依赖它；
        # 写失败时直接忽略，避免可视化影响控制。
        try:
            os.makedirs(os.path.dirname(self.pose_state_path), exist_ok=True)
            state = {
                "step": int(self.step_count),
                "phase": self.phase,
                "test_mode": self.test_mode,
                "test_waypoint_index": int(self.test_waypoint_index),
                "test_waypoint": self.test_waypoints[self.test_waypoint_index].tolist() if self.test_waypoints else None,
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
        # scan_offsets 依次是 120/180/240 度。
        if self.scan_base_heading is None:
            self.scan_base_heading = self.localizer.yaw
        offset = self.scan_offsets[min(self.scan_index, len(self.scan_offsets) - 1)]
        return self._wrap_angle(self.scan_base_heading + offset)

    def _save_scan_frame(self, obs):
        # 保存当前方向的 RGB-D 图像，供后续离线 YOLO / 深度反投影调试。
        # 文件名里带角度偏移，方便把识别结果映射回扫描方向。
        # 现在会保存两类视角：
        # 1. head/video：头部或外部视频相机，视野更大，适合远距离找垃圾；
        # 2. ee：机械臂末端相机，视角低且局部，后续适合靠近后做抓取精定位。
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

    def _save_camera_frame(self, image_cls, camera_name: str, rgb, depth, angle_deg: int) -> bool:
        if rgb is None:
            return False

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
        proprio = obs["proprio"].to(self.device)

        expected_dim = 3 + 3 + 3 + 3 + action_dim + action_dim + action_dim

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
        """Map training-time 12D leg action to current env 20D full-body action."""
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
    
    def predicts(self, obs, current_score):
        """Run policy inference and return current-env full-body action."""
        if current_score > 1:
            return {'action': [], 'giveup': True}
        proprio = obs["proprio"].to(self.device)
        self.step_count += 1
        self.current_velocity_commands = self._compute_navigation_command(obs, proprio)
        if self.step_count % 5 == 0:
            self._write_pose_state()
        action_dim = (int(proprio.shape[-1]) - 12) // 3
        policy_obs = self._extract_policy_obs(obs, action_dim)

        with torch.inference_mode():
            action_train = self.policy(policy_obs)

        if not isinstance(action_train, torch.Tensor):
            action_train = torch.as_tensor(
                action_train, device=self.device, dtype=torch.float32
            )

        action_train = action_train.to(device=self.device, dtype=torch.float32)

        if action_train.ndim == 1:
            action_train = action_train.unsqueeze(0)

        action_env = self._map_policy_action_to_env_action(action_train, action_dim)
        action_env = action_env.cpu().numpy().tolist()
        return {'action': action_env, 'giveup': False}

