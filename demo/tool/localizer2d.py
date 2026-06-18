"""TaskB 策略层二维定位器。

这里不依赖 IsaacLab 运行时，只维护一个轻量的 (x, y, yaw)：
1. 初始位姿来自 taskb_map_prior.py 的出生点先验；
2. 每帧用 proprio 中的机体系速度积分做 dead reckoning；
3. 视觉/LiDAR 如果拟合到投放区圆环，就以小权重修正 xy；
4. clamp_to_reasonable_bounds 防止里程计发散后把导航目标带飞。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

try:
    from demo.tool import taskb_map_prior as prior
except ImportError:
    import taskb_map_prior as prior


@dataclass
class Pose2D:
    xy: np.ndarray
    yaw: float


class Localizer2D:
    """Pose estimator for the strategy layer.

    The first version uses proprioceptive dead reckoning plus a fixed map prior.
    Correction hooks are provided for later circle/boundary matching without
    changing solution.py again.
    """

    def __init__(self, dt: float = 0.02):
        self.dt = dt
        self.pose = Pose2D(xy=prior.SPAWN_XY.copy(), yaw=float(prior.SPAWN_YAW))
        self.last_correction = "spawn_prior"

    @property
    def pose_xy(self) -> np.ndarray:
        return self.pose.xy

    @property
    def yaw(self) -> float:
        return self.pose.yaw

    def reset(self):
        self.pose = Pose2D(xy=prior.SPAWN_XY.copy(), yaw=float(prior.SPAWN_YAW))
        self.last_correction = "spawn_prior"

    def predict_from_proprio(self, proprio):
        """从 proprio 的机体系速度积分位姿。

        注意这里使用的是策略层估计，不是仿真真值；误差会靠投放区观测逐步拉回。
        """

        if proprio is None or proprio.numel() < 6:
            return
        # proprio 前 3 维是 base 线速度，3:6 是 base 角速度，均按机体系理解。
        base_lin_vel_b = proprio[0, 0:3].detach().cpu().numpy().astype(np.float64)
        base_ang_vel_b = proprio[0, 3:6].detach().cpu().numpy().astype(np.float64)

        yaw = prior.wrap_angle(self.pose.yaw + float(base_ang_vel_b[2]) * self.dt)
        c = math.cos(yaw)
        s = math.sin(yaw)
        vx_w = c * base_lin_vel_b[0] - s * base_lin_vel_b[1]
        vy_w = s * base_lin_vel_b[0] + c * base_lin_vel_b[1]
        self.pose = Pose2D(
            xy=self.pose.xy + np.array([vx_w, vy_w], dtype=np.float64) * self.dt,
            yaw=yaw,
        )

    def correct_with_world_drop_circle(self, observed_center_xy: Iterable[float], weight: float = 0.35):
        """Correct pose with a world-frame drop-circle observation.

        This hook is useful once the drop-zone circle is detected in world/map
        coordinates.  The correction shifts the robot pose by the residual
        between observed and prior circle centers.
        """

        observed = np.asarray(observed_center_xy, dtype=np.float64)
        residual = prior.DROP_CENTER_XY - observed
        self.pose = Pose2D(
            xy=self.pose.xy + residual * float(np.clip(weight, 0.0, 1.0)),
            yaw=self.pose.yaw,
        )
        self.last_correction = "drop_circle"

    def correct_yaw_with_boundary(self, observed_boundary_yaw: float, map_boundary_yaw: float, weight: float = 0.25):
        """Correct yaw using a matched boundary-line direction."""

        err = prior.wrap_angle(map_boundary_yaw - observed_boundary_yaw)
        self.pose = Pose2D(
            xy=self.pose.xy,
            yaw=prior.wrap_angle(self.pose.yaw + err * float(np.clip(weight, 0.0, 1.0))),
        )
        self.last_correction = "boundary_yaw"



    def correct_with_drop_estimate(self, estimate, max_center_error: float = 1.5, weight: float = 0.25) -> bool:
        """用世界系投放区圆拟合结果修正位姿。

        只有圆心、半径都接近静态先验时才修正，避免橙色杂物误检导致定位跳变。
        """

        if estimate is None or not hasattr(estimate, "center_xy"):
            return False
        observed = np.asarray(estimate.center_xy, dtype=np.float64)
        center_error = float(np.linalg.norm(observed - prior.DROP_CENTER_XY))
        if center_error > max_center_error:
            return False
        if hasattr(estimate, "radius") and abs(float(estimate.radius) - prior.DROP_RADIUS) > 0.35:
            return False
        self.correct_with_world_drop_circle(observed, weight=weight)
        self.last_correction = getattr(estimate, "source", "drop_estimate")
        return True

    def inspect_extero_for_lidar(self, extero) -> dict:
        """Return conservative diagnostics for the extero height-scan observation.

        The public observation currently exposes a processed height_scan, not raw
        ray-hit coordinates or ranges.  Without exact ordering/semantics, using
        it for pose correction is risky.  This method keeps the interface ready
        and provides simple health metrics for debugging.
        """

        if extero is None:
            return {"available": False, "reason": "missing"}
        try:
            data = extero.detach().cpu().numpy().reshape(-1).astype(np.float64)
        except Exception:
            return {"available": False, "reason": "unreadable"}
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return {"available": False, "reason": "no_finite"}
        return {
            "available": True,
            "count": int(finite.size),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "mean": float(np.mean(finite)),
        }

    def stand_pose_for_drop(self, stand_distance: float | None = None):
        """Compute the preferred stand-off point outside the drop circle."""

        distance = prior.DROP_STAND_DISTANCE if stand_distance is None else stand_distance
        center = prior.DROP_CENTER_XY
        direction = self.pose.xy - center
        norm = np.linalg.norm(direction)
        if norm < 0.25:
            direction = -center
            norm = np.linalg.norm(direction)
        if norm < 1e-6:
            direction = np.array([1.0, 0.0], dtype=np.float64)
        else:
            direction = direction / norm
        pos_xy = center + direction * max(distance, prior.DROP_RADIUS + 0.5)
        face = center - pos_xy
        heading = math.atan2(float(face[1]), float(face[0]))
        return pos_xy, heading

    def clamp_to_reasonable_bounds(self, margin: float = 3.0):
        """Keep severe dead-reckoning failures from exploding navigation."""

        # 只做兜底裁剪，不追求精确地图约束；否则误差较大时容易把状态机卡死。
        bounds = prior.MAP_BOUNDS
        self.pose.xy[0] = np.clip(self.pose.xy[0], bounds["xmin"] - margin, bounds["xmax"] + margin)
        self.pose.xy[1] = np.clip(self.pose.xy[1], bounds["ymin"] - margin, bounds["ymax"] + margin)
