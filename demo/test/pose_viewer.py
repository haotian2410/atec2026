"""TaskB 策略层二维定位/视觉伺服可视化。

运行方式：在 play_atec_task.py 或比赛脚本运行时，另开终端执行：

    python demo/test/pose_viewer.py

重要说明：这里显示的 robot 位置不是仿真真值，也不是直接从 Isaac 读取的
root pose。它读取的是 demo/solution.py 写出的 outputs/taskb_pose_state.json，
其中 pose 来自 Localizer2D：出生点先验 + proprio 速度积分，并用投放区
橙色矮墙 RGB-D / LiDAR height_scan 圆环观测做小权重校正。

这个脚本只用于调试，不会被比赛策略自动 import。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

def configure_matplotlib(gui: bool = False):
    try:
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
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle
    except Exception as err:
        raise SystemExit(
            f"failed to import matplotlib: {err}\n"
            "Try running inside the project conda env, e.g. `conda activate env_isaaclab`, "
            "or install a matplotlib/numpy pair compatible with your Python."
        ) from err
    return plt, Circle, Rectangle

def find_project_root(start: Path | None = None) -> Path:
    start = (start or Path(__file__)).resolve()
    for candidate in (start if start.is_dir() else start.parent, *start.parents):
        if (candidate / "demo").is_dir() and (candidate / "source").is_dir():
            return candidate
    return Path.cwd().resolve()


def resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


PROJECT_ROOT = find_project_root()
DEFAULT_STATE = Path("outputs") / "taskb_pose_state.json"
DEFAULT_SAVE = Path("outputs") / "pose_viewer.png"


def load_state(path: Path):
    # solution.py 每 5 步原子写一次 JSON；读失败通常只是写入瞬间，下一帧会恢复。
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="Project root. Defaults to auto-detected ATEC2026 root.")
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--save", type=Path, default=DEFAULT_SAVE, help="Save viewer image path.")
    parser.add_argument("--gui", action="store_true", help="Open an interactive matplotlib window instead of saving once.")
    parser.add_argument("--watch-save", action="store_true", help="Continuously refresh --save image without opening a GUI.")
    args = parser.parse_args()

    root = find_project_root(args.root)
    state_path = resolve_path(args.state, root)
    save_path = resolve_path(args.save, root)

    plt, Circle, Rectangle = configure_matplotlib(args.gui)
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6), dpi=96)
    if args.gui and hasattr(fig.canvas.manager, "set_window_title"):
        fig.canvas.manager.set_window_title("TaskB 2D Pose Viewer")

    while True:
        state = load_state(state_path)
        ax.clear()
        ax.set_title("TaskB maintained pose (not simulator ground truth)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("x / m")
        ax.set_ylabel("y / m")

        bounds = {"xmin": -10.0, "xmax": 10.0, "ymin": -10.0, "ymax": 10.0}
        drop = {"center_xy": [-3.0, -10.0], "radius": 1.0}
        pose = None
        phase = "waiting"
        correction = None
        lidar = None
        prior_errors = []
        trash = None
        if state:
            # 状态文件由 solution.py::_write_pose_state 生成：
            # map/drop 是静态先验，pose 是 Localizer2D 维护的策略层估计位姿。
            bounds.update(state.get("map_bounds", {}))
            drop.update(state.get("drop_zone", {}))
            pose = state.get("pose")
            phase = state.get("phase", phase)
            correction = state.get("visual_correction")
            lidar = state.get("lidar_diag")
            prior_errors = state.get("prior_errors", []) or []
            trash = state.get("trash")

        width = bounds["xmax"] - bounds["xmin"]
        height = bounds["ymax"] - bounds["ymin"]
        ax.add_patch(Rectangle((bounds["xmin"], bounds["ymin"]), width, height, fill=False, lw=2.0, ec="black"))

        cx, cy = drop["center_xy"]
        radius = drop["radius"]
        ax.add_patch(Circle((cx, cy), radius, fill=False, lw=2.0, ec="tab:orange"))
        ax.plot([cx], [cy], marker="x", color="tab:orange")
        ax.text(cx + 0.1, cy + 0.1, "drop", color="tab:orange")

        if pose:
            # 这里的 x/y/yaw 是“策略认为自己在哪里”：
            # proprio dead-reckoning 积分为主，投放区圆环观测只做保守校正。
            x, y = pose["xy"]
            yaw = pose["yaw"]
            ax.plot([x], [y], marker="o", color="tab:blue", markersize=8)
            ax.arrow(x, y, 0.8 * math.cos(yaw), 0.8 * math.sin(yaw), head_width=0.18, color="tab:blue")
            ax.text(x + 0.15, y + 0.15, "robot", color="tab:blue")

        current_target = None
        if isinstance(trash, dict):
            current_target = trash.get("current_target")
        if isinstance(current_target, dict) and current_target.get("world_xy"):
            # 垃圾位置同样是策略层估计：来自 YOLO 框 + 深度 + 当前估计位姿/FK，
            # 用于观察目标选择是否从 ee 粗定位切到 head/video 精定位。
            tx, ty = current_target["world_xy"]
            ax.plot([tx], [ty], marker="*", color="tab:red", markersize=13)
            ax.text(tx + 0.12, ty + 0.12, "trash", color="tab:red")

        ax.set_xlim(bounds["xmin"] - 2.0, bounds["xmax"] + 2.0)
        ax.set_ylim(bounds["ymin"] - 2.0, bounds["ymax"] + 2.0)
        info = f"phase: {phase}"
        if correction:
            info += f"\nvisual: {correction.get('source')} inliers={correction.get('inlier_count')}"
        if lidar:
            info += (
                f"\nlidar: wall={lidar.get('wall_candidates')} "
                f"inliers={lidar.get('circle_inliers')} "
                f"err={lidar.get('circle_center_error')} "
                f"used={lidar.get('correction_used', False)}"
            )
        if isinstance(trash, dict):
            target = trash.get("current_target")
            info += f"\ntrash_targets: {trash.get('target_count', 0)}"
            if isinstance(target, dict):
                err = target.get("image_error") or [0.0, 0.0]
                info += (
                    f"\ntarget: {target.get('frame_kind', 'scan')}:{target.get('camera')} {target.get('label')} "
                    f"conf={target.get('confidence'):.2f} dist={target.get('distance_hint_m'):.2f}"
                    f" err=({err[0]:.2f},{err[1]:.2f})"
                )
            if trash.get("servo"):
                servo = trash.get("servo")
                info += (
                    f"\nservo: mode={servo.get('mode')} "
                    f"vx={servo.get('lin_x'):.2f} vy={servo.get('lin_y'):.2f} "
                    f"wz={servo.get('yaw_rate'):.2f} ready={servo.get('ready_to_grasp')}"
                )
        if prior_errors:
            info += "\nprior_errors: " + "; ".join(str(e) for e in prior_errors[:2])
        ax.text(0.02, 0.98, info, transform=ax.transAxes, va="top", fontsize=8, family="DejaVu Sans Mono")

        if not args.gui:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=96, bbox_inches="tight")
            print(save_path, flush=True)
            if not args.watch_save:
                break
            time.sleep(args.interval)
            continue

        try:
            if not plt.fignum_exists(fig.number):
                break
            fig.canvas.draw_idle()
            plt.pause(args.interval)
        except RuntimeError as err:
            print(f"matplotlib draw failed: {err}")
            print("use non-GUI mode: python demo/test/pose_viewer.py --watch-save")
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
