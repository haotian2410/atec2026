"""查看并绘制 solution.py 写出的 LiDAR height_scan 定位诊断。

本脚本不读取仿真真值，只读取两个调试文件：
1. outputs/taskb_pose_state.json：当前策略状态和 lidar_diag 数字诊断；
2. outputs/taskb_lidar_debug/latest_lidar_debug.npz：最近一次 height_scan 单帧反推伪点云。

注意：这里的点云不是传感器直接给出的 xyz/range，而是根据 IsaacLab
mdp.height_scan = sensor_z - ray_hit_z - offset 和固定射线方向近似反推的
单帧伪点云。它用于判断“这帧是否能看出投放区矮墙圆环”，不是多帧融合地图。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _configure_matplotlib(gui: bool = False):
    import matplotlib
    if gui:
        matplotlib.use("TkAgg", force=True)
    else:
        matplotlib.use("Agg", force=True)
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "figure.dpi": 96,
    })
    return matplotlib

def find_project_root(start: Path | None = None) -> Path:
    start = (start or Path(__file__)).resolve()
    for candidate in (start if start.is_dir() else start.parent, *start.parents):
        if (candidate / "demo").is_dir() and (candidate / "source").is_dir():
            return candidate
    return Path.cwd().resolve()


def resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


ROOT = find_project_root()
STATE = Path("outputs") / "taskb_pose_state.json"
DEBUG_NPZ = Path("outputs") / "taskb_lidar_debug" / "latest_lidar_debug.npz"
DEFAULT_SAVE = Path("outputs") / "lidar_debug.png"


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def print_state(state_path: Path):
    # STATE 由 solution.py::_write_pose_state 周期写出；不存在说明策略还没跑到写状态。
    if not state_path.exists():
        print("no state file", state_path)
        return
    state = _load_json(state_path)
    if state is None:
        print("state unreadable", state_path)
        return
    print("phase:", state.get("phase"))
    print("pose:", state.get("pose"))
    print("lidar_diag:")
    for k, v in (state.get("lidar_diag") or {}).items():
        print(f"  {k}: {v}")
    print("visual_correction:", state.get("visual_correction"))


def plot_lidar_debug(npz_path: Path, save_path: Path | None = None, gui: bool = False):
    if not npz_path.exists():
        print("no lidar debug npz", npz_path)
        return

    _configure_matplotlib(gui)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    data = np.load(npz_path, allow_pickle=False)
    world_points = data["world_points"]
    wall_points = data["wall_points"]
    fit_center = data["fit_center"]
    fit_radius = float(data["fit_radius"][0])
    prior_center = data["prior_center"]
    prior_radius = float(data["prior_radius"][0])
    pose_xy = data["pose_xy"]
    yaw = float(data["yaw"][0])
    diag = json.loads(str(data["diag_json"]))

    fig, ax = plt.subplots(figsize=(7, 7), dpi=96)
    ax.set_title("TaskB LiDAR height_scan pseudo points (single frame)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("world x / m")
    ax.set_ylabel("world y / m")

    if world_points.size:
        z = world_points[:, 2]
        sc = ax.scatter(world_points[:, 0], world_points[:, 1], c=z, s=5, cmap="viridis", alpha=0.45, label="all pseudo hits")
        fig.colorbar(sc, ax=ax, label="estimated hit z / m")
    if wall_points.size:
        ax.scatter(wall_points[:, 0], wall_points[:, 1], s=16, c="tab:red", label="wall z candidates")

    ax.add_patch(Circle(tuple(prior_center), prior_radius, fill=False, lw=2.0, ec="tab:orange", label="prior drop circle"))
    ax.plot([prior_center[0]], [prior_center[1]], marker="x", color="tab:orange")

    if np.isfinite(fit_center).all() and math.isfinite(fit_radius):
        ax.add_patch(Circle(tuple(fit_center), fit_radius, fill=False, lw=2.0, ec="tab:green", label="fitted circle"))
        ax.plot([fit_center[0]], [fit_center[1]], marker="+", color="tab:green", markersize=12)

    ax.plot([pose_xy[0]], [pose_xy[1]], marker="o", color="tab:blue", markersize=8, label="estimated robot pose")
    ax.arrow(pose_xy[0], pose_xy[1], 0.8 * math.cos(yaw), 0.8 * math.sin(yaw), head_width=0.18, color="tab:blue")

    info = (
        f"count: {diag.get('count')} used: {diag.get('used')}\n"
        f"wall_candidates: {diag.get('wall_candidates')} inliers: {diag.get('circle_inliers')}\n"
        f"center_error: {diag.get('circle_center_error')} radius: {diag.get('circle_radius')}"
    )
    ax.text(0.02, 0.98, info, transform=ax.transAxes, va="top", fontsize=8, family="DejaVu Sans Mono")
    ax.legend(loc="lower right", fontsize=8)
    if not gui:
        if save_path is None:
            save_path = resolve_path(DEFAULT_SAVE, ROOT)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=96, bbox_inches="tight")
        print(save_path)
    else:
        try:
            plt.show()
        except RuntimeError as err:
            print(f"matplotlib draw failed: {err}")
            print("use non-GUI mode: python demo/test/lidar_height_scan_check.py --plot")


def main():
    parser = argparse.ArgumentParser(description="Inspect or plot TaskB LiDAR height_scan debug output.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Project root. Defaults to auto-detected ATEC2026 root.")
    parser.add_argument("--state", type=Path, default=STATE)
    parser.add_argument("--npz", type=Path, default=DEBUG_NPZ)
    parser.add_argument("--plot", action="store_true", help="Plot latest single-frame pseudo point cloud and fitted circle.")
    parser.add_argument("--save", type=Path, default=DEFAULT_SAVE, help="Save plot image path.")
    parser.add_argument("--gui", action="store_true", help="Open an interactive matplotlib window instead of saving image.")
    args = parser.parse_args()

    root = find_project_root(args.root)
    state_path = resolve_path(args.state, root)
    npz_path = resolve_path(args.npz, root)
    save_path = resolve_path(args.save, root)

    print_state(state_path)
    if args.plot:
        plot_lidar_debug(npz_path, save_path=save_path, gui=args.gui)


if __name__ == "__main__":
    main()
