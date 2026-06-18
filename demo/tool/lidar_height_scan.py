"""从 IsaacLab height_scan 中恢复保守二维地标。

mdp.height_scan 暴露的是 sensor_z - ray_hit_z - offset，不是原始 xyz 点云。
由于 TaskB 的雷达射线图案固定，可以根据射线方向和高度值反推一个近似
局部命中点，再投到世界系尝试拟合投放区圆环。该结果只做小权重定位校正。
"""

from __future__ import annotations

import numpy as np

try:
    from demo.tool import taskb_map_prior as prior
    from demo.tool.drop_zone import fit_circle_ransac
except ImportError:
    import taskb_map_prior as prior
    from drop_zone import fit_circle_ransac


HEIGHT_SCAN_OFFSET = 0.5


def lidar_ray_directions() -> np.ndarray:
    # 根据静态 LiDAR 配置展开每条射线的单位方向，顺序需与 height_scan 保持一致。
    cfg = prior.LIDAR_PRIOR
    v_angles = np.linspace(cfg["vertical_fov_range"][0], cfg["vertical_fov_range"][1], cfg["channels"])
    num_h = int(np.ceil((cfg["horizontal_fov_range"][1] - cfg["horizontal_fov_range"][0]) / cfg["horizontal_res_deg"]) + 1)
    h_angles = np.linspace(cfg["horizontal_fov_range"][0], cfg["horizontal_fov_range"][1], num_h)
    if abs(abs(cfg["horizontal_fov_range"][0] - cfg["horizontal_fov_range"][1]) - 360.0) < 1e-6:
        h_angles = h_angles[:-1]
    vv, hh = np.meshgrid(np.deg2rad(v_angles), np.deg2rad(h_angles), indexing="ij")
    x = np.cos(vv) * np.cos(hh)
    y = np.cos(vv) * np.sin(hh)
    z = np.sin(vv)
    return np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float64)


RAY_DIRECTIONS = lidar_ray_directions()


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


def height_scan_to_local_points(extero, ground_z: float = 0.0) -> tuple[np.ndarray | None, dict]:
    """Convert height_scan to conservative local hit points.

    Returns local points in the LiDAR/base horizontal frame and diagnostics.
    """

    arr = _to_numpy(extero)
    if arr is None:
        return None, {"available": False, "reason": "missing"}
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    finite_mask = np.isfinite(arr)
    values = arr[finite_mask]
    if values.size == 0:
        return None, {"available": False, "reason": "no_finite"}

    dirs = RAY_DIRECTIONS
    n = min(values.size, dirs.shape[0])
    values = values[:n]
    dirs = dirs[:n]

    # 多数射线命中平地：h = sensor_z - ground_z - 0.5。
    # 墙体命中会拉低数值，所以用较高分位估计 sensor_z，比中位数更稳。
    ground_like = values[np.isfinite(values)]
    sensor_z_est = float(np.percentile(ground_like, 80.0) + HEIGHT_SCAN_OFFSET + ground_z)
    hit_z = sensor_z_est - values - HEIGHT_SCAN_OFFSET

    dz = dirs[:, 2]
    valid = np.abs(dz) > 0.05
    t = (hit_z - sensor_z_est) / dz
    valid &= np.isfinite(t) & (t > 0.05) & (t < prior.LIDAR_PRIOR["max_distance"] + 0.5)
    local = dirs[:, :2] * t[:, None]
    points = local[valid]
    hit_z_valid = hit_z[valid]

    diag = {
        "available": True,
        "count": int(values.size),
        "used": int(points.shape[0]),
        "sensor_z_est": sensor_z_est,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "expected_rays": int(RAY_DIRECTIONS.shape[0]),
    }
    if points.shape[0] == 0:
        return None, diag
    return np.column_stack([points, hit_z_valid]), diag


def local_points_to_world(points_local_z: np.ndarray, pose_xy: np.ndarray, yaw: float) -> np.ndarray:
    points = np.asarray(points_local_z, dtype=np.float64)
    c = np.cos(yaw)
    s = np.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    xy_w = (rot @ points[:, :2].T).T + np.asarray(pose_xy, dtype=np.float64)
    return np.column_stack([xy_w, points[:, 2]])


def estimate_drop_from_height_scan(extero, pose_xy: np.ndarray, yaw: float):
    """Fit the drop-zone wall from height_scan pseudo points, if visible."""

    local_points, diag = height_scan_to_local_points(extero)
    if local_points is None:
        return None, diag
    world_points = local_points_to_world(local_points, pose_xy, yaw)

    # 只保留疑似矮墙点：地面 z 接近 0，投放区墙体侧面/顶面大致落在 0.08..0.75。
    z = world_points[:, 2]
    wall_mask = np.isfinite(z) & (z > 0.08) & (z < 0.75)
    xy = world_points[wall_mask, :2]
    diag["wall_candidates"] = int(xy.shape[0])
    if xy.shape[0] < 20:
        return None, diag

    estimate = fit_circle_ransac(xy, radius_range=(0.70, 1.35), iterations=96, inlier_thresh=0.18)
    if estimate is None:
        return None, diag
    estimate.source = "lidar_height_scan_world"
    center_error = float(np.linalg.norm(estimate.center_xy - prior.DROP_CENTER_XY))
    diag["circle_center_error"] = center_error
    diag["circle_radius"] = float(estimate.radius)
    diag["circle_inliers"] = int(estimate.inlier_count)
    if center_error > 1.75:
        return None, diag
    if abs(estimate.radius - prior.DROP_RADIUS) > 0.40:
        return None, diag
    if estimate.inlier_count < 20:
        return None, diag
    return estimate, diag


def debug_height_scan_projection(extero, pose_xy: np.ndarray, yaw: float) -> dict:
    """返回单帧 height_scan 反推结果，供调试脚本画图。

    注意：这里输出的是“当前帧 + 当前策略层位姿”下的伪点云，不是多帧累计点云。
    solution.py 会把这些数组保存成 npz，demo/test/lidar_height_scan_check.py 再画出来。
    """

    local_points, diag = height_scan_to_local_points(extero)
    if local_points is None:
        return {
            "available": False,
            "diag": diag,
            "local_points": np.zeros((0, 3), dtype=np.float64),
            "world_points": np.zeros((0, 3), dtype=np.float64),
            "wall_points": np.zeros((0, 2), dtype=np.float64),
            "fit_center": np.array([np.nan, np.nan], dtype=np.float64),
            "fit_radius": np.nan,
        }

    world_points = local_points_to_world(local_points, pose_xy, yaw)
    z = world_points[:, 2]
    wall_mask = np.isfinite(z) & (z > 0.08) & (z < 0.75)
    wall_points = world_points[wall_mask, :2]

    fit_center = np.array([np.nan, np.nan], dtype=np.float64)
    fit_radius = np.nan
    if wall_points.shape[0] >= 20:
        estimate = fit_circle_ransac(wall_points, radius_range=(0.70, 1.35), iterations=96, inlier_thresh=0.18)
        if estimate is not None:
            fit_center = np.asarray(estimate.center_xy, dtype=np.float64)
            fit_radius = float(estimate.radius)
            diag = dict(diag)
            diag["circle_center_error"] = float(np.linalg.norm(fit_center - prior.DROP_CENTER_XY))
            diag["circle_radius"] = fit_radius
            diag["circle_inliers"] = int(estimate.inlier_count)

    diag = dict(diag)
    diag["wall_candidates"] = int(wall_points.shape[0])
    return {
        "available": True,
        "diag": diag,
        "local_points": local_points,
        "world_points": world_points,
        "wall_points": wall_points,
        "fit_center": fit_center,
        "fit_radius": fit_radius,
    }
