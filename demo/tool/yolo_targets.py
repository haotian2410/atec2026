"""TaskB YOLO 目标数据桥接和视觉伺服工具。

本模块不直接跑 YOLO，只定义“检测程序 -> solution.py 状态机”的 JSON 契约，
并把检测框、深度、扫描角度转换为排序后的 TrashTarget。这样检测模型可以
由队友单独替换，主控制循环只需要轮询 JSON。

主要职责：
- load_yolo_targets(): 从本次 scan_dir 的 yolo_results.json 读取检测结果；
- fill_depth_from_saved_images(): 从同名 depth.npy 补齐 bbox 中心深度；
- detections_to_targets(): 将 bbox/depth 转为 TrashTarget，并估计 point_body/world_xy；
- servo_command_from_target(): 将当前目标转换为底盘速度。远距离 ee 用粗接近，
  近距离 head/body 以横移和前后为主、少旋转，并把垃圾调到本体视角中下部。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from demo.tool import taskb_map_prior as prior
    from demo.tool.piper_kinematics import DEFAULT_ARM_QPOS, PiperKinematics
except ImportError:
    import taskb_map_prior as prior
    from piper_kinematics import DEFAULT_ARM_QPOS, PiperKinematics


# solution.py 会优先读本次 scan_dir 下的 JSON；这些是全局兜底路径。
YOLO_RESULT_CANDIDATES = (
    "outputs/taskb_yolo_results.json",
    "outputs/taskb_scan/latest_yolo_results.json",
)


@dataclass
class YoloDetection:
    camera: str
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float = 1.0
    label: str = "trash"
    image_size: tuple[int, int] = (prior.IMAGE_WIDTH, prior.IMAGE_HEIGHT)
    depth_m: float | None = None
    scan_angle_deg: float | None = None
    source_image: str | None = None
    point_camera: tuple[float, float, float] | None = None
    frame_kind: str = "scan"

    @property
    def center_px(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    @property
    def bbox_area_frac(self) -> float:
        w, h = self.image_size
        x1, y1, x2, y2 = self.bbox_xyxy
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        return float(area / max(float(w * h), 1.0))

    @property
    def image_error(self) -> tuple[float, float]:
        """返回归一化图像偏差；+x 表示目标在图像中心右侧。"""

        w, h = self.image_size
        cx, cy = self.center_px
        return (
            float((cx - 0.5 * w) / max(0.5 * w, 1.0)),
            float((cy - 0.5 * h) / max(0.5 * h, 1.0)),
        )


@dataclass
class TrashTarget:
    target_id: str
    camera: str
    label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    image_size: tuple[int, int]
    image_error: tuple[float, float]
    depth_m: float | None
    has_valid_depth: bool
    distance_hint_m: float
    world_xy: tuple[float, float] | None
    scan_angle_deg: float | None
    source_image: str | None
    point_camera: tuple[float, float, float] | None
    point_body: tuple[float, float, float] | None
    frame_kind: str
    state: str = "rough"

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "camera": self.camera,
            "label": self.label,
            "confidence": float(self.confidence),
            "bbox_xyxy": list(self.bbox_xyxy),
            "image_size": list(self.image_size),
            "image_error": list(self.image_error),
            "depth_m": None if self.depth_m is None else float(self.depth_m),
            "has_valid_depth": bool(self.has_valid_depth),
            "distance_hint_m": float(self.distance_hint_m),
            "world_xy": None if self.world_xy is None else list(self.world_xy),
            "scan_angle_deg": self.scan_angle_deg,
            "source_image": self.source_image,
            "point_camera": None if self.point_camera is None else list(self.point_camera),
            "point_body": None if self.point_body is None else list(self.point_body),
            "frame_kind": self.frame_kind,
            "state": self.state,
        }


def load_yolo_targets(project_root: str | os.PathLike, scan_dir: str | os.PathLike | None = None,
                      pose_xy: Iterable[float] | None = None, yaw: float = 0.0,
                      arm_qpos: Iterable[float] | None = None,
                      scan_base_yaw: float | None = None) -> tuple[list[TrashTarget], dict[str, Any]]:
    """读取队友 YOLO 检测 JSON，并转换成按优先级排序的 TrashTarget。

    支持 JSON 形态：
      1. {"detections": [{"camera": "head", "bbox": [x1,y1,x2,y2], ...}]}
      2. {"head": [...], "ee": [...]}，每个元素里包含 bbox。

    可选字段包括 confidence/conf、label/class/name、image_size/[width,height]、
    depth/depth_m、point_camera/point_cam_xyz、scan_angle_deg/angle_deg、source_image。

    如果没有 depth_m 但 source_image 指向保存过的 RGB 图，会尝试读取匹配的
    *_depth.npy，并在 bbox 中心附近取深度中位数。
    """

    root = Path(project_root)
    candidates = [root / p for p in YOLO_RESULT_CANDIDATES]
    if scan_dir:
        scan_path = Path(scan_dir)
        candidates.insert(0, scan_path / "yolo_results.json")
        candidates.insert(1, scan_path / "detections.json")

    result_path = next((p for p in candidates if p.exists()), None)
    if result_path is None:
        return [], {"available": False, "reason": "missing_yolo_json", "checked": [str(p) for p in candidates]}

    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as err:
        return [], {"available": False, "reason": f"read_error: {err}", "path": str(result_path)}

    meta = data.get("meta", {}) if isinstance(data, dict) else {}
    updated_at = meta.get("updated_at") if isinstance(meta, dict) else None
    try:
        updated_at = None if updated_at is None else float(updated_at)
    except Exception:
        updated_at = None

    detections = parse_detections(data)
    detections = fill_depth_from_saved_images(detections, result_path.parent, root)
    targets = detections_to_targets(
        detections,
        pose_xy=pose_xy,
        yaw=yaw,
        arm_qpos=arm_qpos,
        scan_base_yaw=scan_base_yaw,
    )
    return targets, {
        "available": True,
        "path": str(result_path),
        "detections": len(detections),
        "targets": len(targets),
        "updated_at": updated_at,
        "age_s": None if updated_at is None else max(0.0, time.time() - updated_at),
    }


def parse_detections(data: Any) -> list[YoloDetection]:
    if isinstance(data, dict) and isinstance(data.get("detections"), list):
        raw_items = data["detections"]
    elif isinstance(data, dict):
        raw_items = []
        for camera in ("head", "video", "body", "ee", "end_effector"):
            for item in data.get(camera, []) or []:
                if isinstance(item, dict):
                    copied = dict(item)
                    copied.setdefault("camera", camera)
                    raw_items.append(copied)
    elif isinstance(data, list):
        raw_items = data
    else:
        raw_items = []

    detections: list[YoloDetection] = []
    for item in raw_items:
        det = _parse_one_detection(item)
        if det is not None:
            detections.append(det)
    return detections


def detections_to_targets(detections: list[YoloDetection], pose_xy: Iterable[float] | None = None,
                          yaw: float = 0.0, arm_qpos: Iterable[float] | None = None,
                          scan_base_yaw: float | None = None) -> list[TrashTarget]:
    pose = None if pose_xy is None else np.asarray(pose_xy, dtype=np.float64)
    targets: list[TrashTarget] = []
    for index, det in enumerate(detections):
        point_camera = det.point_camera or _pixel_depth_to_camera_point(det)
        point_body = _camera_point_to_body(det.camera, point_camera, arm_qpos)
        distance = _distance_hint(det, point_body)
        world_xy = _estimate_world_xy(det, distance, pose, yaw, point_body, scan_base_yaw=scan_base_yaw)
        state = "precise" if _is_head_camera(det.camera) else "rough"
        targets.append(
            TrashTarget(
                target_id=f"{det.camera}_{index:02d}",
                camera=det.camera,
                label=det.label,
                confidence=det.confidence,
                bbox_xyxy=det.bbox_xyxy,
                image_size=det.image_size,
                image_error=det.image_error,
                depth_m=det.depth_m,
                has_valid_depth=det.depth_m is not None,
                distance_hint_m=distance,
                world_xy=world_xy,
                scan_angle_deg=det.scan_angle_deg,
                source_image=det.source_image,
                point_camera=point_camera,
                point_body=point_body,
                frame_kind=det.frame_kind,
                state=state,
            )
        )

    # 接近阶段优先级：
    # 1. live head/video/body 中最近目标，用于近距离校验和姿态微调；
    # 2. live ee 目标，用末端相机深度/FK 做粗靠近；
    # 3. scan head/ee 目标，作为实时检测暂时没更新时的兜底。
    targets.sort(key=lambda t: (_target_priority(t), t.distance_hint_m, -t.confidence))
    return targets


def servo_command_from_target(target: TrashTarget) -> dict[str, float | bool | str]:
    """根据目标框偏差生成保守的底盘速度指令。

    行走策略使用机器人本体系速度：+x 前进、+y 向左、+yaw 逆时针。
    远距离 ee 目标用于粗靠近；近距离 head/body 目标用于抓取预备位微调。
    近距离阶段不追求画面中心，而是让垃圾落在 head 视角中下部：
    - 横向误差主要用 lin_y 修正，目标在图像右侧时向右横移；
    - 距离和竖直图像误差共同决定 lin_x；
    - yaw_rate 只保留很小修正，避免旋转造成目标框跳动和定位误差放大。
    """

    err_x, err_y = target.image_error
    distance = target.distance_hint_m
    close = target.camera in ("head", "video", "body") and distance < 1.20

    target_err_y = None
    vertical_error = None
    if close:
        # 本体相机近距离微调以横移和前后为主，尽量少转。目标点设在画面中下部，
        # 这样垃圾处在低位 head 相机里更接近抓取预备视角。
        target_err_y = 0.48
        vertical_error = err_y - target_err_y
        forward = float(np.clip(0.28 * (distance - 0.72) - 0.12 * vertical_error, -0.10, 0.20))
        lateral = float(np.clip(-0.42 * err_x, -0.22, 0.22))
        yaw_rate = float(np.clip(-0.08 * err_x, -0.05, 0.05))
    else:
        forward = float(np.clip(0.55 * (distance - 0.9), 0.0, 0.75))
        lateral = float(np.clip(-0.35 * err_x, -0.35, 0.35))
        yaw_rate = float(np.clip(-0.25 * err_x, -0.22, 0.22))

    hold_for_grasp = (
        target.camera in ("head", "video", "body")
        and target.has_valid_depth
        and abs(err_x) < 0.72
        and 0.24 <= err_y <= 1.05
        and 0.52 <= distance <= 1.12
    )
    ready = hold_for_grasp and abs(err_x) < 0.22 and 0.30 <= err_y <= 0.98
    if ready:
        forward = lateral = yaw_rate = 0.0
    elif hold_for_grasp:
        # Hold is a slow refinement mode, not a state-machine stop condition.
        forward = float(np.clip(forward, -0.05, 0.05))
        lateral = float(np.clip(lateral, -0.08, 0.08))
        yaw_rate = float(np.clip(yaw_rate, -0.02, 0.02))

    return {
        "lin_x": forward,
        "lin_y": lateral,
        "yaw_rate": yaw_rate,
        "ready_to_grasp": bool(ready),
        "hold_for_grasp": bool(hold_for_grasp),
        "mode": "head_precise" if close else "ee_or_far_rough",
        "target_err_y": target_err_y,
        "vertical_error": vertical_error,
    }


def _parse_one_detection(item: Any) -> YoloDetection | None:
    if not isinstance(item, dict):
        return None
    bbox = item.get("bbox") or item.get("bbox_xyxy") or item.get("xyxy")
    if bbox is None or len(bbox) != 4:
        return None

    camera = str(item.get("camera") or item.get("cam") or "head").lower()
    if camera in ("end_effector", "eef", "wrist"):
        camera = "ee"
    if camera == "body":
        camera = "head"

    image_size = item.get("image_size") or item.get("size")
    if image_size is None:
        image_size = (prior.IMAGE_WIDTH, prior.IMAGE_HEIGHT)
    if len(image_size) != 2:
        image_size = (prior.IMAGE_WIDTH, prior.IMAGE_HEIGHT)
    image_size = (int(image_size[0]), int(image_size[1]))

    bbox_xyxy = tuple(float(v) for v in bbox)
    if not _bbox_is_usable(bbox_xyxy, image_size):
        return None

    depth = item.get("depth_m", item.get("depth", None))
    try:
        depth = None if depth is None else float(depth)
        if depth is not None and (not math.isfinite(depth) or depth <= 0.0):
            depth = None
    except Exception:
        depth = None

    point_camera = _parse_point_camera(item)

    return YoloDetection(
        camera=camera,
        bbox_xyxy=bbox_xyxy,
        confidence=float(item.get("confidence", item.get("conf", 1.0))),
        label=str(item.get("label", item.get("class", item.get("name", "trash")))),
        image_size=image_size,
        depth_m=depth,
        scan_angle_deg=_optional_float(item.get("scan_angle_deg", item.get("angle_deg"))),
        source_image=item.get("source_image"),
        point_camera=point_camera,
        frame_kind=str(item.get("frame_kind") or _infer_frame_kind(item.get("source_image"))).lower(),
    )


def _bbox_is_usable(bbox: tuple[float, float, float, float], image_size: tuple[int, int]) -> bool:
    # YOLO 有时会给出几乎铺满整张图、并贴住多个边界的大框。
    # 这种框的中心和深度都不适合做抓取目标，会导致二维图里的垃圾点大幅跳变。
    w, h = image_size
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return False
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    area_frac = (bw * bh) / max(float(w * h), 1.0)
    touch = 0
    margin = 3.0
    touch += int(x1 <= margin)
    touch += int(y1 <= margin)
    touch += int(x2 >= w - margin)
    touch += int(y2 >= h - margin)
    if area_frac > 0.75:
        return False
    if area_frac > 0.35 and touch >= 2:
        return False
    return True


def _distance_hint(det: YoloDetection, point_body: tuple[float, float, float] | None = None) -> float:
    if point_body is not None:
        return float(np.clip(np.linalg.norm(np.asarray(point_body, dtype=np.float64)[:2]), 0.2, 8.0))
    if det.depth_m is not None:
        return float(np.clip(det.depth_m, 0.2, 8.0))
    # 没有深度时用框面积粗估距离：框越小越远。末端相机先按更远处理。
    area = max(det.bbox_area_frac, 1e-4)
    nominal = 0.55 / math.sqrt(area)
    if det.camera == "ee":
        nominal *= 1.25
    return float(np.clip(nominal, 0.5, 6.0))


def _estimate_world_xy(det: YoloDetection, distance: float, pose: np.ndarray | None,
                       base_yaw: float, point_body: tuple[float, float, float] | None = None,
                       scan_base_yaw: float | None = None) -> tuple[float, float] | None:
    if pose is None:
        return None
    # scan 图是在历史朝向拍的，不能用当前 yaw 解释图像误差或 body 点。
    # scan_base_yaw 是“面向投放区”的扫描基准角，scan_angle_deg 是保存图片时
    # 相对该基准角的 120/180/240 度偏移。live 图没有 scan_angle_deg，直接用当前 yaw。
    scan_yaw = base_yaw
    if det.scan_angle_deg is not None:
        base = base_yaw if scan_base_yaw is None else float(scan_base_yaw)
        scan_yaw = prior.wrap_angle(base + math.radians(det.scan_angle_deg))

    err_x, _err_y = det.image_error
    if point_body is not None and det.frame_kind != "live":
        pb = np.asarray(point_body, dtype=np.float64)
        c = math.cos(scan_yaw)
        s = math.sin(scan_yaw)
        world = pose + np.array([c * pb[0] - s * pb[1], s * pb[0] + c * pb[1]], dtype=np.float64)
        return (float(world[0]), float(world[1]))

    # live 目标的二维显示用与图像伺服一致的近似方位，避免 ee FK/外参误差
    # 让可视化目标点和实际移动方向严重不一致。
    forward = distance
    lateral_left = -err_x * distance * 0.65
    c = math.cos(scan_yaw)
    s = math.sin(scan_yaw)
    world = pose + np.array(
        [c * forward - s * lateral_left, s * forward + c * lateral_left],
        dtype=np.float64,
    )
    return (float(world[0]), float(world[1]))


def _target_priority(target: TrashTarget) -> int:
    live = target.frame_kind == "live"
    if live and _is_head_camera(target.camera):
        return 0
    if live and target.camera == "ee":
        return 1
    if _is_head_camera(target.camera):
        return 2
    return 3


def _is_head_camera(camera: str) -> bool:
    return camera in ("head", "video", "body")


def _infer_frame_kind(source_image: Any) -> str:
    if not source_image:
        return "scan"
    name = Path(str(source_image)).name
    return "live" if name.startswith("live_") else "scan"


def _optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        value = float(value)
        return value if math.isfinite(value) else None
    except Exception:
        return None



def fill_depth_from_saved_images(detections: list[YoloDetection], json_dir: Path, project_root: Path) -> list[YoloDetection]:
    """Fill missing depth_m from saved *_depth.npy files when possible.

    The detector can return only bbox + source_image.  For a saved image named
    scan_01_180deg_head_rgb.png, this looks for scan_01_180deg_head_depth.npy
    and averages a 7x7 finite-depth window around the bbox center.
    """

    filled: list[YoloDetection] = []
    for det in detections:
        if det.depth_m is not None or not det.source_image:
            filled.append(det)
            continue
        depth_path = _matching_depth_path(det.source_image, json_dir, project_root)
        depth = _mean_depth_at_bbox_center(depth_path, det.bbox_xyxy) if depth_path else None
        if depth is None:
            filled.append(det)
            continue
        filled.append(
            YoloDetection(
                camera=det.camera,
                bbox_xyxy=det.bbox_xyxy,
                confidence=det.confidence,
                label=det.label,
                image_size=det.image_size,
                depth_m=depth,
                scan_angle_deg=det.scan_angle_deg,
                source_image=det.source_image,
                point_camera=det.point_camera,
                frame_kind=det.frame_kind,
            )
        )
    return filled


def _parse_point_camera(item: dict[str, Any]) -> tuple[float, float, float] | None:
    point = item.get("point_camera") or item.get("point_cam") or item.get("point_cam_xyz") or item.get("xyz_camera")
    if point is None or len(point) != 3:
        return None
    try:
        xyz = tuple(float(v) for v in point)
    except Exception:
        return None
    if not all(math.isfinite(v) for v in xyz):
        return None
    return xyz


def _pixel_depth_to_camera_point(det: YoloDetection) -> tuple[float, float, float] | None:
    if det.depth_m is None:
        return None
    camera = prior.CAMERA_PRIORS.get("ee" if det.camera == "ee" else "head", prior.HEAD_CAMERA)
    z = float(det.depth_m)
    u, v = det.center_px
    x = (u - prior.camera_cx(camera)) / max(prior.camera_fx(camera), 1e-6) * z
    y = (v - prior.camera_cy(camera)) / max(prior.camera_fy(camera), 1e-6) * z
    # 统一给上层使用的相机点约定：x 向前，y 向左，z 向上/垂直近似。
    # RGB-D 常见 pinhole 输出是横向、纵向、深度，这里转成策略层近似前左上。
    return (z, -x, -y)


def _camera_point_to_body(camera_name: str, point_camera: tuple[float, float, float] | None,
                          arm_qpos: Iterable[float] | None = None) -> tuple[float, float, float] | None:
    if point_camera is None:
        return None
    point = np.asarray(point_camera, dtype=np.float64).reshape(1, 3)
    if camera_name == "ee":
        q = np.asarray(arm_qpos, dtype=np.float64) if arm_qpos is not None else DEFAULT_ARM_QPOS
        if q.size < 6:
            q = DEFAULT_ARM_QPOS
        kin = PiperKinematics()
        gripper_from_base = kin.fk(q[:6])
        cam = prior.EE_CAMERA
        r_cam_gripper = prior.quat_to_mat_wxyz(cam["quat_parent_wxyz"])
        t_cam_gripper = np.asarray(cam["pos_parent"], dtype=np.float64)
        point_gripper = (r_cam_gripper @ point.T + t_cam_gripper.reshape(3, 1)).reshape(3)
        point_body = gripper_from_base[:3, :3] @ point_gripper + gripper_from_base[:3, 3]
        return tuple(float(v) for v in point_body)

    world_like = prior.camera_points_to_world(point, prior.HEAD_CAMERA, np.array([0.0, 0.0]), 0.0)[0]
    return tuple(float(v) for v in world_like)


def _matching_depth_path(source_image: str, json_dir: Path, project_root: Path) -> Path | None:
    src = Path(source_image)
    candidates = []
    if src.is_absolute():
        candidates.append(src)
    else:
        candidates.extend([json_dir / src, project_root / src])
    for image_path in candidates:
        name = image_path.name
        depth_names = []
        if name.endswith("_rgb.png"):
            depth_names.append(name.replace("_rgb.png", "_depth.npy"))
        depth_names.append(image_path.stem + "_depth.npy")
        for depth_name in depth_names:
            depth_path = image_path.with_name(depth_name)
            if depth_path.exists():
                return depth_path
    return None


def _mean_depth_at_bbox_center(depth_path: Path | None, bbox: tuple[float, float, float, float], radius: int = 3) -> float | None:
    if depth_path is None:
        return None
    try:
        depth = np.load(depth_path)
    except Exception:
        return None
    if depth.ndim == 3:
        depth = depth[..., 0]
    h, w = depth.shape[:2]
    x1, y1, x2, y2 = bbox
    cx = int(round(0.5 * (x1 + x2)))
    cy = int(round(0.5 * (y1 + y2)))
    x0, xa = max(0, cx - radius), min(w, cx + radius + 1)
    y0, ya = max(0, cy - radius), min(h, cy + radius + 1)
    patch = np.asarray(depth[y0:ya, x0:xa], dtype=np.float64)
    finite = patch[np.isfinite(patch) & (patch > 0.05) & (patch < 10.0)]
    if finite.size == 0:
        return None
    return float(np.median(finite))
