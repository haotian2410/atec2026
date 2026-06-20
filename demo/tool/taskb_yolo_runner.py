"""TaskB 扫描图片的 YOLO 运行脚手架。

该脚本刻意放在 solution.py 外部：控制循环只读取 yolo_results.json，检测可以
在另一个终端或队友程序里跑，避免 YOLO 推理阻塞 Isaac 控制步。

典型用法：

    python demo/tool/taskb_yolo_runner.py \
        --scan-dir outputs/taskb_scan/20260617_210400 \
        --model /path/to/best.pt

如果省略 --model，会写出空检测结果，用于先跑通文件接口。若不用 Ultralytics，
替换 TaskBYoloRunner._load_model() / _predict_one_image() 即可。
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any

import numpy as np


SCAN_IMAGE_RE = re.compile(r"scan_(?P<index>\d+)_(?P<angle>\d+)deg_(?P<camera>head|video|ee)_rgb\.png$")
LIVE_IMAGE_RE = re.compile(r"live_(?P<step>\d+)_(?P<camera>head|video|ee)_rgb\.png$")


class TaskBYoloRunner:
    """读取 TaskB 保存的扫描图，运行检测器并写出 yolo_results.json。

    输出字段会被 demo/tool/yolo_targets.py 消费：
      camera, bbox, confidence, label, source_image, scan_angle_deg, depth_m。
    """

    def __init__(self, model_path: str | os.PathLike | None = None, conf: float = 0.25,
                 head_conf: float = 0.45, head_dedup: bool = True):
        self.model_path = None if model_path is None else Path(model_path)
        self.conf = float(conf)
        self.head_conf = float(head_conf)
        self.head_dedup = bool(head_dedup)
        self.model = self._load_model(self.model_path)
        self._prediction_cache: dict[str, dict[str, Any]] = {}

    def run_scan_dir(self, scan_dir: str | os.PathLike, output: str | os.PathLike | None = None,
                     max_live_per_camera: int = 1) -> dict[str, Any]:
        scan_path = Path(scan_dir)
        detections: list[dict[str, Any]] = []
        predicted_count = 0
        cached_count = 0

        image_paths = self._candidate_images(scan_path, max_live_per_camera=max_live_per_camera)
        for image_path in image_paths:
            meta = self._parse_image_name(image_path)
            if meta is None or not self._image_ready(image_path):
                continue
            try:
                image_detections, from_cache = self._predict_one_image_cached(image_path)
            except Exception as err:
                # solution.py 可能正在写 PNG，OpenCV 读到空/半截文件会报错。
                # watch 模式不能因为单张图失败退出，下一轮文件稳定后会重新检测。
                print(json.dumps({"skip_image": str(image_path), "reason": repr(err)}, ensure_ascii=False), flush=True)
                continue
            if from_cache:
                cached_count += 1
            else:
                predicted_count += 1
            if meta["camera"] in ("head", "video") and self.head_dedup:
                image_detections = self._dedup_detections(image_detections)

            for det in image_detections:
                bbox = det.get("bbox")
                if bbox is None or len(bbox) != 4:
                    continue
                confidence = float(det.get("confidence", det.get("conf", 1.0)))
                min_conf = self.head_conf if meta["camera"] in ("head", "video") else self.conf
                if confidence < min_conf:
                    continue

                record = {
                    "camera": meta["camera"],
                    "bbox": [float(v) for v in bbox],
                    "confidence": confidence,
                    "label": str(det.get("label", det.get("name", "trash"))),
                    "source_image": image_path.name,
                    "frame_kind": meta["frame_kind"],
                }
                if meta.get("angle_deg") is not None:
                    record["scan_angle_deg"] = float(meta["angle_deg"])
                depth = self._depth_for_bbox(image_path, record["bbox"])
                if depth is not None:
                    record["depth_m"] = depth
                detections.append(record)

        result = {
            "detections": detections,
            "meta": {
                "scan_dir": str(scan_path),
                "model": None if self.model_path is None else str(self.model_path),
                "confidence_threshold": self.conf,
                "head_confidence_threshold": self.head_conf,
                "head_dedup": self.head_dedup,
                "count": len(detections),
                "image_count": len(image_paths),
                "predicted_image_count": predicted_count,
                "cached_image_count": cached_count,
                "updated_at": time.time(),
            },
        }
        output_path = Path(output) if output else scan_path / "yolo_results.json"
        self._write_json_atomic(output_path, result)
        return result

    def watch_scan_dir(self, scan_dir: str | os.PathLike | None = None, output: str | os.PathLike | None = None,
                       interval: float = 0.25, max_live_per_camera: int = 1):
        # 持续检测扫描目录。若 scan_dir=None，每轮都重新选择最新 outputs/taskb_scan/*，
        # 这样主任务重启生成新时间戳目录后，runner 不会继续盯着旧目录写 JSON。
        # 接近阶段 solution.py 会不断写 live_* 当前帧；这里每轮只取每个相机最新
        # live 图，避免旧实时帧干扰目标选择。
        last_scan_path = None
        while True:
            scan_path = Path(scan_dir) if scan_dir is not None else latest_scan_dir()
            if scan_path is None:
                print(json.dumps({"available": False, "reason": "missing_scan_dir"}, ensure_ascii=False), flush=True)
                time.sleep(max(float(interval), 0.05))
                continue
            if scan_path != last_scan_path:
                print(json.dumps({"switch_scan_dir": str(scan_path)}, ensure_ascii=False), flush=True)
                last_scan_path = scan_path
            result = self.run_scan_dir(scan_path, output=output, max_live_per_camera=max_live_per_camera)
            print(json.dumps(result["meta"], ensure_ascii=False), flush=True)
            time.sleep(max(float(interval), 0.05))

    def _load_model(self, model_path: Path | None):
        # 模型位置留在这里：
        # 1. 如果你们用 Ultralytics YOLO，直接传 --model best.pt 即可。
        # 2. 如果你们用自己的 PyTorch/ONNX/TensorRT 推理，把加载逻辑写在这里，
        #    并让 _predict_one_image() 返回 bbox 列表。
        if model_path is None:
            return None
        try:
            from ultralytics import YOLO
        except Exception as err:
            raise RuntimeError(
                "Ultralytics is not available. Install it or replace "
                "TaskBYoloRunner._load_model/_predict_one_image with your detector."
            ) from err
        return YOLO(str(model_path))

    def _predict_one_image_cached(self, image_path: Path) -> tuple[list[dict[str, Any]], bool]:
        # scan_* 图片不会再变化，live_* 图片按新文件名滚动；用 mtime/size 做缓存
        # 可以避免 watch 模式每 0.25s 重复推理同一批 scan 图。
        try:
            stat = image_path.stat()
        except OSError:
            return [], False
        cache_key = str(image_path.resolve())
        signature = (stat.st_mtime_ns, stat.st_size, self.conf, self.head_conf, self.head_dedup)
        cached = self._prediction_cache.get(cache_key)
        if cached is not None and cached.get("signature") == signature:
            return list(cached.get("detections", [])), True

        detections = self._predict_one_image(image_path)
        self._prediction_cache[cache_key] = {
            "signature": signature,
            "detections": [dict(det) for det in detections],
        }
        return detections, False

    def _predict_one_image(self, image_path: Path) -> list[dict[str, Any]]:
        # 无模型时输出空检测，便于先跑通完整文件接口，不影响 solution.py。
        if self.model is None:
            return []

        # 默认实现：Ultralytics YOLO。
        # 返回格式统一成：
        #   {"bbox": [x1, y1, x2, y2], "confidence": 0.9, "label": "trash"}
        results = self.model.predict(str(image_path), conf=self.conf, verbose=False)
        parsed: list[dict[str, Any]] = []
        for result in results:
            names = getattr(result, "names", {}) or {}
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy = boxes.xyxy.detach().cpu().numpy()
            conf = boxes.conf.detach().cpu().numpy()
            cls = boxes.cls.detach().cpu().numpy().astype(int)
            for box, score, cls_id in zip(xyxy, conf, cls):
                parsed.append(
                    {
                        "bbox": [float(v) for v in box],
                        "confidence": float(score),
                        "label": str(names.get(int(cls_id), "trash")),
                    }
                )
        return parsed

    @staticmethod
    def _bbox_area(box: list[float]) -> float:
        x1, y1, x2, y2 = [float(v) for v in box]
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @staticmethod
    def _bbox_iou(a: list[float], b: list[float]) -> float:
        ax1, ay1, ax2, ay2 = [float(v) for v in a]
        bx1, by1, bx2, by2 = [float(v) for v in b]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = TaskBYoloRunner._bbox_area([ix1, iy1, ix2, iy2])
        denom = TaskBYoloRunner._bbox_area(a) + TaskBYoloRunner._bbox_area(b) - inter
        return inter / denom if denom > 1e-9 else 0.0

    @staticmethod
    def _bbox_center_distance_norm(a: list[float], b: list[float], width: float = 640.0, height: float = 480.0) -> float:
        acx, acy = 0.5 * (float(a[0]) + float(a[2])), 0.5 * (float(a[1]) + float(a[3]))
        bcx, bcy = 0.5 * (float(b[0]) + float(b[2])), 0.5 * (float(b[1]) + float(b[3]))
        return math.hypot((acx - bcx) / max(width, 1.0), (acy - bcy) / max(height, 1.0))

    @staticmethod
    def _dedup_detections(detections: list[dict[str, Any]], iou_threshold: float = 0.20,
                          center_threshold: float = 0.22) -> list[dict[str, Any]]:
        # 新 YOLO 在近距离 head 图上会对同一物体给多个相邻框。
        # 同图同类中，IoU 较高或中心非常接近的框只保留最高置信度。
        kept: list[dict[str, Any]] = []
        sorted_dets = sorted(
            detections,
            key=lambda item: float(item.get("confidence", item.get("conf", 0.0))),
            reverse=True,
        )
        for det in sorted_dets:
            bbox = det.get("bbox")
            if bbox is None or len(bbox) != 4:
                continue
            label = str(det.get("label", det.get("name", "trash")))
            duplicate = False
            for old in kept:
                old_label = str(old.get("label", old.get("name", "trash")))
                old_bbox = old.get("bbox")
                if old_label != label or old_bbox is None or len(old_bbox) != 4:
                    continue
                if (
                    TaskBYoloRunner._bbox_iou(bbox, old_bbox) >= iou_threshold
                    or TaskBYoloRunner._bbox_center_distance_norm(bbox, old_bbox) <= center_threshold
                ):
                    duplicate = True
                    break
            if not duplicate:
                kept.append(det)
        return kept

    @staticmethod
    def _image_ready(image_path: Path, min_age_s: float = 0.15, min_size: int = 128) -> bool:
        # 避免读取 solution.py 尚未写完的 PNG。保存 RGB 图不是原子写，
        # 新扫描目录刚出现时最容易读到 0 字节或半截文件。
        try:
            stat = image_path.stat()
        except OSError:
            return False
        if stat.st_size < min_size:
            return False
        return (time.time() - stat.st_mtime) >= min_age_s

    @staticmethod
    def _candidate_images(scan_path: Path, max_live_per_camera: int = 1) -> list[Path]:
        scan_images: list[Path] = []
        live_by_camera: dict[str, list[Path]] = {}
        for image_path in sorted(scan_path.glob("*_rgb.png")):
            meta = TaskBYoloRunner._parse_image_name(image_path)
            if meta is None:
                continue
            if meta["frame_kind"] == "live":
                live_by_camera.setdefault(meta["camera"], []).append(image_path)
            else:
                scan_images.append(image_path)

        live_images: list[Path] = []
        for paths in live_by_camera.values():
            paths = sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)
            live_images.extend(paths[:max_live_per_camera])
        return sorted(scan_images) + sorted(live_images)

    @staticmethod
    def _parse_image_name(image_path: Path) -> dict[str, Any] | None:
        scan_match = SCAN_IMAGE_RE.match(image_path.name)
        if scan_match:
            return {
                "frame_kind": "scan",
                "scan_index": int(scan_match.group("index")),
                "angle_deg": int(scan_match.group("angle")),
                "camera": scan_match.group("camera"),
            }
        live_match = LIVE_IMAGE_RE.match(image_path.name)
        if live_match:
            return {
                "frame_kind": "live",
                "step": int(live_match.group("step")),
                "angle_deg": None,
                "camera": live_match.group("camera"),
            }
        return None

    @staticmethod
    def _write_json_atomic(output_path: Path, result: dict[str, Any]):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, output_path)

    @staticmethod
    def _depth_for_bbox(image_path: Path, bbox: list[float], radius: int = 3) -> float | None:
        depth_path = image_path.with_name(image_path.name.replace("_rgb.png", "_depth.npy"))
        if not depth_path.exists():
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
        value = float(np.median(finite))
        return value if math.isfinite(value) else None


def latest_scan_dir(outputs_dir: str | os.PathLike = "outputs/taskb_scan") -> Path | None:
    # 默认选最近修改的扫描目录，方便机器人刚拍完图后直接运行。
    root = Path(outputs_dir)
    if not root.exists():
        return None
    dirs = [p for p in root.iterdir() if p.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description="Run YOLO on saved TaskB scan images.")
    parser.add_argument("--scan-dir", type=Path, default=None, help="Scan image directory. Defaults to latest outputs/taskb_scan/*.")
    parser.add_argument("--model", type=Path, default=os.environ.get("TASKB_YOLO_MODEL"), help="YOLO model path, e.g. best.pt.")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path. Defaults to <scan-dir>/yolo_results.json.")
    parser.add_argument("--conf", type=float, default=0.25, help="Base confidence threshold for non-head cameras.")
    parser.add_argument("--head-conf", type=float, default=0.45, help="Confidence threshold for head/video cameras.")
    parser.add_argument("--no-head-dedup", action="store_true", help="Disable duplicate-box suppression for head/video cameras.")
    parser.add_argument("--watch", action="store_true", help="Keep detecting new live_* frames during approach.")
    parser.add_argument("--interval", type=float, default=0.25, help="Watch polling interval in seconds.")
    parser.add_argument("--max-live-per-camera", type=int, default=12, help="Number of recent live frames per camera to keep in each watch result.")
    args = parser.parse_args()

    runner = TaskBYoloRunner(model_path=args.model, conf=args.conf, head_conf=args.head_conf, head_dedup=not args.no_head_dedup)
    if args.watch:
        # watch 模式下不指定 --scan-dir 时会自动跟随最新扫描目录。
        runner.watch_scan_dir(args.scan_dir, output=args.output, interval=args.interval, max_live_per_camera=args.max_live_per_camera)
    else:
        scan_dir = args.scan_dir or latest_scan_dir()
        if scan_dir is None:
            raise SystemExit("No scan directory found under outputs/taskb_scan.")
        result = runner.run_scan_dir(scan_dir, output=args.output, max_live_per_camera=args.max_live_per_camera)
        print(json.dumps(result["meta"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
