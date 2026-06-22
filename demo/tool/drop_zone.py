"""TaskB 投放区感知工具。

目标是从 RGB-D 或伪点云中找橙色矮墙圆环，并输出稳定的投放区估计。
本文件只使用 numpy/torch 风格数据转换，不依赖 IsaacLab 运行时，便于
在 solution.py、调试脚本和离线检测流程之间复用。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

try:
    from demo.tool import taskb_map_prior as map_prior
except ImportError:
    import taskb_map_prior as map_prior


TASK_B_BOOTSTRAP_CENTER = np.array([-3.0, -10.0], dtype=np.float64)
TASK_B_BOOTSTRAP_RADIUS = 1.0
TASK_B_MAP_CENTER = np.array([0.0, 0.0], dtype=np.float64)


@dataclass
class DropZoneEstimate:
    center_xy: np.ndarray
    radius: float
    score: float
    source: str
    inlier_count: int = 0


@dataclass
class StandPose:
    pos_xy: np.ndarray
    heading: float


class DropZoneTracker:
    """跨帧维护一个稳定的投放区估计。

    初始值使用 TaskB 地图先验；当视觉候选分数更高时直接替换，
    分数较低但可用时做小权重融合，避免单帧噪声导致圆心抖动。
    """

    def __init__(self):
        self.estimate = DropZoneEstimate(
            center_xy=TASK_B_BOOTSTRAP_CENTER.copy(),
            radius=TASK_B_BOOTSTRAP_RADIUS,
            score=0.1,
            source="task_b_bootstrap",
            inlier_count=0,
        )

    def update(self, candidate: DropZoneEstimate | None) -> DropZoneEstimate:
        if candidate is None:
            return self.estimate
        if candidate.score >= self.estimate.score or self.estimate.source == "task_b_bootstrap":
            self.estimate = candidate
        else:
            alpha = 0.15
            self.estimate = DropZoneEstimate(
                center_xy=(1.0 - alpha) * self.estimate.center_xy + alpha * candidate.center_xy,
                radius=(1.0 - alpha) * self.estimate.radius + alpha * candidate.radius,
                score=max(self.estimate.score, candidate.score),
                source=f"{self.estimate.source}+{candidate.source}",
                inlier_count=max(self.estimate.inlier_count, candidate.inlier_count),
            )
        return self.estimate


def compute_stand_pose(
    estimate: DropZoneEstimate,
    robot_xy: Iterable[float] | None = None,
    stand_distance: float = 1.7,
) -> StandPose:
    """Return a point outside the circle, facing the drop-zone center."""

    center = np.asarray(estimate.center_xy, dtype=np.float64)
    if robot_xy is None:
        direction = TASK_B_MAP_CENTER - center
    else:
        direction = np.asarray(robot_xy, dtype=np.float64) - center
        if np.linalg.norm(direction) < 0.25:
            direction = TASK_B_MAP_CENTER - center
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        direction = np.array([1.0, 0.0], dtype=np.float64)
    else:
        direction = direction / norm

    pos_xy = center + direction * max(stand_distance, estimate.radius + 0.6)
    face = center - pos_xy
    heading = math.atan2(float(face[1]), float(face[0]))
    return StandPose(pos_xy=pos_xy, heading=heading)


def fit_circle_ransac(
    xy: np.ndarray,
    radius_range: tuple[float, float] = (0.75, 1.25),
    iterations: int = 96,
    inlier_thresh: float = 0.08,
    rng_seed: int = 7,
) -> DropZoneEstimate | None:
    """用 RANSAC 从二维点中拟合圆环。

    输入点通常来自橙色墙体 RGB-D 投影或 height_scan 伪点云；
    输出会包含圆心、半径、内点数和用于排序的 score。
    """

    points = np.asarray(xy, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] < 12:
        return None

    rng = np.random.default_rng(rng_seed)
    best = None
    best_mask = None

    for _ in range(iterations):
        ids = rng.choice(points.shape[0], size=3, replace=False)
        circle = _circle_from_three_points(points[ids])
        if circle is None:
            continue
        center, radius = circle
        if not (radius_range[0] <= radius <= radius_range[1]):
            continue
        distances = np.linalg.norm(points - center, axis=1)
        residual = np.abs(distances - radius)
        mask = residual < inlier_thresh
        inlier_count = int(mask.sum())
        if inlier_count < 10:
            continue

        coverage = _angular_coverage(points[mask], center)
        boundary_bonus = _boundary_bonus(center)
        radius_score = 1.0 - min(abs(radius - TASK_B_BOOTSTRAP_RADIUS) / 0.5, 1.0)
        score = inlier_count * 0.03 + coverage * 1.2 + boundary_bonus + radius_score
        if best is None or score > best.score:
            best = DropZoneEstimate(
                center_xy=center,
                radius=float(radius),
                score=float(score),
                source="ransac_xy",
                inlier_count=inlier_count,
            )
            best_mask = mask

    if best is None or best_mask is None:
        return None

    refined = _least_squares_circle(points[best_mask])
    if refined is not None:
        center, radius = refined
        if radius_range[0] <= radius <= radius_range[1]:
            best.center_xy = center
            best.radius = float(radius)
    return best


def orange_depth_points_from_obs(obs: dict) -> np.ndarray | None:
    """Extract approximate orange low-wall points from head/video RGB-D.

    Returns camera-frame points.  Use estimate_world_drop_from_obs() when a
    pose estimate is available and world-frame circle fitting is needed.
    """

    points, _camera_name = orange_depth_points_from_obs_with_camera(obs)
    return points


def _orange_depth_points_from_arrays(rgb, depth, camera_name: str) -> np.ndarray | None:
    rgb_np = _to_numpy(rgb)
    depth_np = _to_numpy(depth)
    if rgb_np is None or depth_np is None:
        return None

    rgb_np = np.asarray(rgb_np)
    depth_np = np.asarray(depth_np)
    if rgb_np.ndim == 4:
        rgb_np = rgb_np[0]
    if depth_np.ndim == 4:
        depth_np = depth_np[0]
    if depth_np.ndim == 3:
        depth_np = depth_np[..., 0]

    red = rgb_np[..., 0].astype(np.int16)
    green = rgb_np[..., 1].astype(np.int16)
    blue = rgb_np[..., 2].astype(np.int16)
    # 投放区矮墙是明显橙色；阈值故意偏保守，少检比误检更安全。
    mask = (red > 140) & (green > 55) & (green < 190) & (blue < 90) & (red > green + 25)
    valid_depth = np.isfinite(depth_np) & (depth_np > 0.2) & (depth_np < 12.0)
    mask &= valid_depth
    if int(mask.sum()) < 50:
        return None

    ys, xs = np.nonzero(mask)
    if ys.shape[0] > 5000:
        step = max(1, ys.shape[0] // 5000)
        ys = ys[::step]
        xs = xs[::step]

    height, width = depth_np.shape
    camera = map_prior.CAMERA_PRIORS.get(camera_name, map_prior.HEAD_CAMERA)
    fx = map_prior.camera_fx(camera)
    fy = map_prior.camera_fy(camera)
    cx = map_prior.camera_cx(camera)
    cy = map_prior.camera_cy(camera)

    z = depth_np[ys, xs].astype(np.float64)
    x = (xs.astype(np.float64) - cx) * z / fx
    y = (ys.astype(np.float64) - cy) * z / fy
    return np.stack([x, y, z], axis=1)




def orange_depth_points_from_obs_with_camera(obs: dict):
    """Return orange RGB-D points and the camera prior name used.

    The existing orange_depth_points_from_obs() keeps the old camera-frame API.
    This helper adds the camera name so callers can transform head/video points
    into the world frame when a pose estimate is available.
    """

    image = obs.get("image") if isinstance(obs, dict) else None
    if not isinstance(image, dict):
        return None, None

    rgb = image.get("head_rgb")
    depth = image.get("head_depth")
    camera_name = "head"
    if rgb is None or depth is None:
        # Do not reuse head extrinsics for video frames. If video needs to drive
        # localization, add an explicit video camera prior first.
        return None, None

    points = _orange_depth_points_from_arrays(rgb, depth, camera_name)
    return points, camera_name


def estimate_world_drop_from_obs(obs: dict, pose_xy: np.ndarray, yaw: float) -> DropZoneEstimate | None:
    """Estimate the drop-circle center in world coordinates from RGB-D.

    This is intentionally conservative.  It only uses base-fixed head/video
    camera data and returns None unless the fitted circle looks like the known
    TaskB drop-zone circle.
    """

    points_cam, camera_name = orange_depth_points_from_obs_with_camera(obs)
    if points_cam is None or points_cam.shape[0] < 30:
        return None
    camera = map_prior.CAMERA_PRIORS.get(camera_name)
    if camera is None or camera.get("parent_link") != "base_link":
        return None

    points_w = map_prior.camera_points_to_world(points_cam, camera, np.asarray(pose_xy, dtype=np.float64), float(yaw))
    z = points_w[:, 2]
    wall_mask = np.isfinite(z) & (z > 0.03) & (z < 0.75)
    xy = points_w[wall_mask, :2]
    if xy.shape[0] < 20:
        return None

    estimate = fit_circle_ransac(xy, radius_range=(0.70, 1.30), iterations=128, inlier_thresh=0.12)
    if estimate is None:
        return None
    estimate.source = f"{camera_name}_rgbd_world"

    # 再用静态地图先验做门控：橙色阈值可能捡到别的物体，
    # 相机坐标约定也可能有细微偏差，不能让错误观测直接改定位。
    center_error = float(np.linalg.norm(estimate.center_xy - map_prior.DROP_CENTER_XY))
    if center_error > 1.5:
        return None
    if abs(estimate.radius - map_prior.DROP_RADIUS) > 0.35:
        return None
    if estimate.inlier_count < 20:
        return None
    estimate.score += max(0.0, 1.5 - center_error)
    return estimate

def estimate_from_obs(obs: dict) -> DropZoneEstimate | None:
    """Best-effort drop-zone estimate from currently available observations."""

    points = orange_depth_points_from_obs(obs)
    if points is None:
        return None

    # In camera coordinates the wall circle may be tilted in the image.  For the
    # first debug version, fit in lateral-depth space to recover a stable ring.
    xy = points[:, [0, 2]]
    estimate = fit_circle_ransac(xy, radius_range=(0.65, 1.35), inlier_thresh=0.10)
    if estimate is None:
        return None
    estimate.source = "orange_rgbd_camera"
    return estimate


def _to_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _circle_from_three_points(points: np.ndarray):
    p1, p2, p3 = points
    a = p2 - p1
    b = p3 - p1
    denom = 2.0 * (a[0] * b[1] - a[1] * b[0])
    if abs(denom) < 1e-8:
        return None
    aa = float(np.dot(a, a))
    bb = float(np.dot(b, b))
    center = p1 + np.array(
        [
            (b[1] * aa - a[1] * bb) / denom,
            (a[0] * bb - b[0] * aa) / denom,
        ],
        dtype=np.float64,
    )
    radius = float(np.linalg.norm(center - p1))
    if not np.isfinite(radius):
        return None
    return center, radius


def _least_squares_circle(points: np.ndarray):
    if points.shape[0] < 3:
        return None
    x = points[:, 0]
    y = points[:, 1]
    a = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    b = x * x + y * y
    try:
        sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    center = sol[:2]
    c = sol[2]
    radius_sq = float(np.dot(center, center) + c)
    if radius_sq <= 0.0:
        return None
    return center.astype(np.float64), math.sqrt(radius_sq)


def _angular_coverage(points: np.ndarray, center: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    bins = np.floor((angles + math.pi) / (2.0 * math.pi) * 36.0).astype(np.int32)
    bins = np.clip(bins, 0, 35)
    return float(np.unique(bins).shape[0] / 36.0)


def _boundary_bonus(center: np.ndarray) -> float:
    # Task B terrain is 20m x 20m in the source config.  A drop zone near the
    # edge is preferred, but the score is intentionally soft because frame
    # transforms may shift local observations.
    distance_to_edge = min(abs(10.0 - center[0]), abs(-10.0 - center[0]), abs(10.0 - center[1]), abs(-10.0 - center[1]))
    return float(max(0.0, 1.0 - distance_to_edge / 4.0))
