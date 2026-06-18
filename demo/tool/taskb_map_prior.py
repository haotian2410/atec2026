"""TaskB 静态地图和传感器先验。

本文件刻意不引入 IsaacLab 运行时，只保存从公开任务配置中整理出来的常量：
出生点、投放区圆心、地图边界、相机内外参和 LiDAR 扫描参数。策略层可在
评测时直接使用这些先验做导航、坐标变换和观测门控。
"""

from __future__ import annotations

import math
import numpy as np


# -----------------------------
# 1. Static TaskB map landmarks
# -----------------------------

# TaskB-B2Piper 出生点先验，来自 env_cfg.py。
SPAWN_XY = np.array([-10.0, -10.0], dtype=np.float64)
SPAWN_YAW = 0.0

# 奖励/终止条件使用的投放区目标，来自 task_b/env_cfg.py。
DROP_CENTER_XY = np.array([-3.0, -10.0], dtype=np.float64)
DROP_RADIUS = 1.0
DROP_STAND_DISTANCE = 1.5

# Terrain generator size is 20m x 20m, but TaskB's generated terrain tile is
# effectively centered at world (-10, -10): the terrain-local ring at (+7, 0)
# appears in task rewards as world (-3, -10).  Therefore world bounds are
# approximately [-20, 0] x [-20, 0], and the B2 spawn (-10, -10) is map center.
MAP_ORIGIN_XY = np.array([-10.0, -10.0], dtype=np.float64)
MAP_SIZE_XY = np.array([20.0, 20.0], dtype=np.float64)
MAP_BOUNDS = {
    "xmin": -20.0,
    "xmax": 0.0,
    "ymin": -20.0,
    "ymax": 0.0,
}

# 地图边界线格式：n dot p = d，其中 n 指向地图内侧。
BOUNDARY_LINES = [
    {"name": "west", "normal": np.array([1.0, 0.0], dtype=np.float64), "d": MAP_BOUNDS["xmin"]},
    {"name": "east", "normal": np.array([-1.0, 0.0], dtype=np.float64), "d": -MAP_BOUNDS["xmax"]},
    {"name": "south", "normal": np.array([0.0, 1.0], dtype=np.float64), "d": MAP_BOUNDS["ymin"]},
    {"name": "north", "normal": np.array([0.0, -1.0], dtype=np.float64), "d": -MAP_BOUNDS["ymax"]},
]

# Sanity checks derived from TaskB source:
# terrain local drop center (+7, 0) + MAP_ORIGIN_XY == reward center (-3, -10).
TERRAIN_LOCAL_DROP_CENTER_XY = np.array([7.0, 0.0], dtype=np.float64)


# -----------------------------
# 2. Camera intrinsics/extrinsics
# -----------------------------

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
HORIZONTAL_APERTURE = 20.955

HEAD_CAMERA = {
    "name": "head",
    "width": IMAGE_WIDTH,
    "height": IMAGE_HEIGHT,
    "focal_length": 24.0,
    "horizontal_aperture": HORIZONTAL_APERTURE,
    # Offset of the head camera sensor frame relative to B2 base_link, from
    # UNITREE_B2_CFG.head_camera_offset.  Quaternion is scalar-first wxyz.
    "parent_link": "base_link",
    "pos_parent": np.array([0.4216099977493286, 0.02500000037252903, 0.06185099855065346], dtype=np.float64),
    "quat_parent_wxyz": np.array([0.9659258262890683, 0.0, 0.25881904510252074, 0.0], dtype=np.float64),
    "convention": "world",
}

EE_CAMERA = {
    "name": "ee",
    "width": IMAGE_WIDTH,
    "height": IMAGE_HEIGHT,
    "focal_length": 15.0,
    "horizontal_aperture": HORIZONTAL_APERTURE,
    # Offset of the end-effector camera relative to gripper_base, from
    # UNITREE_B2_PIPER_CFG.ee_camera_offset.  Its parent is moving, so this is
    # only a local prior; world projection needs arm FK or runtime link pose.
    "parent_link": "gripper_base",
    "pos_parent": np.array([-0.05, 0.0, 0.06], dtype=np.float64),
    "quat_parent_wxyz": np.array([0.7071067811865476, 0.0, 0.0, -0.7071067811865475], dtype=np.float64),
    "convention": "ros",
}

CAMERA_PRIORS = {
    "head": HEAD_CAMERA,
    "ee": EE_CAMERA,
}


# -----------------------------
# 3. LiDAR / height scan prior
# -----------------------------

LIDAR_PRIOR = {
    "parent_link": "base_link",
    "horizontal_fov_range": (-180.0, 180.0),
    "vertical_fov_range": (-20.0, 20.0),
    "horizontal_res_deg": 1.0,
    "channels": 16,
    "max_distance": 10.0,
    "update_period": 0.1,
}


def camera_fx(camera: dict) -> float:
    return camera["focal_length"] / camera["horizontal_aperture"] * camera["width"]


def camera_fy(camera: dict) -> float:
    return camera_fx(camera)


def camera_cx(camera: dict) -> float:
    return camera["width"] * 0.5


def camera_cy(camera: dict) -> float:
    return camera["height"] * 0.5


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


# -----------------------------
# 4. Rigid transform helpers
# -----------------------------

def quat_to_mat_wxyz(q) -> np.ndarray:
    """Convert scalar-first quaternion (w, x, y, z) to a 3x3 matrix."""

    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = q / max(np.linalg.norm(q), 1e-12)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotz(yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def camera_points_to_world(points_cam: np.ndarray, camera: dict, base_xy: np.ndarray, base_yaw: float) -> np.ndarray:
    """用固定相机先验把相机系点变换到世界系。

    这是策略层近似变换：head/video 相机固定在 base_link 上，可以直接用
    base_xy/base_yaw 投影；ee 相机父节点是 gripper_base，若没有机械臂 FK，
    不应直接用这个函数做世界投影。
    """

    points_cam = np.asarray(points_cam, dtype=np.float64)
    r_base_w = rotz(base_yaw)
    t_base_w = np.array([base_xy[0], base_xy[1], 0.0], dtype=np.float64)
    r_cam_parent = quat_to_mat_wxyz(camera["quat_parent_wxyz"])
    t_cam_parent = np.asarray(camera["pos_parent"], dtype=np.float64)
    return (r_base_w @ (r_cam_parent @ points_cam.T + t_cam_parent.reshape(3, 1))).T + t_base_w



def validate_taskb_prior() -> list[str]:
    """返回地图/传感器先验的可读一致性错误。"""

    errors: list[str] = []
    expected_drop = MAP_ORIGIN_XY + TERRAIN_LOCAL_DROP_CENTER_XY
    if np.linalg.norm(expected_drop - DROP_CENTER_XY) > 1e-6:
        errors.append(f"drop center mismatch: terrain {expected_drop} vs reward {DROP_CENTER_XY}")
    if not (MAP_BOUNDS["xmin"] <= SPAWN_XY[0] <= MAP_BOUNDS["xmax"]):
        errors.append(f"spawn x outside map bounds: {SPAWN_XY[0]}")
    if not (MAP_BOUNDS["ymin"] <= SPAWN_XY[1] <= MAP_BOUNDS["ymax"]):
        errors.append(f"spawn y outside map bounds: {SPAWN_XY[1]}")
    if not (MAP_BOUNDS["xmin"] <= DROP_CENTER_XY[0] <= MAP_BOUNDS["xmax"]):
        errors.append(f"drop x outside map bounds: {DROP_CENTER_XY[0]}")
    if not (MAP_BOUNDS["ymin"] <= DROP_CENTER_XY[1] <= MAP_BOUNDS["ymax"]):
        errors.append(f"drop y outside map bounds: {DROP_CENTER_XY[1]}")
    return errors
