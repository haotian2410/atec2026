#!/usr/bin/env python3
"""Inspect Task-D staged reset poses without running PPO training."""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect Task-D reset local/world poses.")
parser.add_argument("--task", type=str, default="ATEC-TaskD-RL-B2-Climb-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=1, help="Number of zero-action steps after reset.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--terrain_x", type=float, default=12.0)
parser.add_argument("--terrain_y", type=float, default=8.0)
parser.add_argument("--terrain_origin_x", type=float, default=1.8)
parser.add_argument("--terrain_origin_y", type=float, default=4.0)
parser.add_argument("--debug_reset_positions", action="store_true", default=False)
parser.add_argument("--debug_reset_num_envs", type=int, default=8)
parser.add_argument("--probe_local_x", type=float, default=None, help="Move robot to this local x and step once to probe local-x rewards/terminations.")
parser.add_argument("--max_print_envs", type=int, default=16, help="Print at most this many envs, plus the last few when truncated.")
parser.add_argument("--print_all_envs", action="store_true", default=False, help="Print every env pose.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import atec_rl_lab.tasks  # noqa: F401, E402
from atec_rl_lab.tasks.task_d.rl_env_cfg import sync_task_d_terrain_grid  # noqa: E402


def _print_pose_table(env, terrain_x: float, terrain_y: float, terrain_origin_x: float, terrain_origin_y: float):
    unwrapped = env.unwrapped
    origins = unwrapped.scene.env_origins
    robot = unwrapped.scene["robot"]
    box = unwrapped.scene["box"]
    robot_world = robot.data.root_pos_w
    box_world = box.data.root_pos_w
    robot_local = robot_world - origins
    box_local = box_world - origins
    x_min = -float(terrain_origin_x)
    x_max = float(terrain_x) - float(terrain_origin_x)
    y_min = -float(terrain_origin_y)
    y_max = float(terrain_y) - float(terrain_origin_y)

    print("[TaskD inspect] env_count=", unwrapped.num_envs)
    print("[TaskD inspect] scene.env_spacing=", getattr(unwrapped.cfg.scene, "env_spacing", None), flush=True)
    terrain_generator = getattr(unwrapped.cfg.scene.terrain, "terrain_generator", None)
    if terrain_generator is not None:
        print(
            "[TaskD inspect] terrain rows/cols=",
            getattr(terrain_generator, "num_rows", None),
            getattr(terrain_generator, "num_cols", None),
            flush=True,
        )
    print("[TaskD inspect] expected local terrain bounds:", f"x=[{x_min}, {x_max}]", f"y=[{y_min}, {y_max}]", flush=True)

    all_env_ids = list(range(unwrapped.num_envs))
    if args_cli.print_all_envs or unwrapped.num_envs <= args_cli.max_print_envs:
        env_ids_to_print = all_env_ids
    else:
        head_count = max(args_cli.max_print_envs, 0)
        tail_count = min(4, max(unwrapped.num_envs - head_count, 0))
        env_ids_to_print = all_env_ids[:head_count] + all_env_ids[-tail_count:]
        skipped = unwrapped.num_envs - len(env_ids_to_print)
        print(f"[TaskD inspect] omitted {skipped} env rows; pass --print_all_envs to show all.", flush=True)

    for env_id in env_ids_to_print:
        r_local = robot_local[env_id]
        b_local = box_local[env_id]
        robot_inside = (r_local[0] >= x_min) and (r_local[0] <= x_max) and (r_local[1] >= y_min) and (r_local[1] <= y_max)
        box_inside = (b_local[0] >= x_min) and (b_local[0] <= x_max) and (b_local[1] >= y_min) and (b_local[1] <= y_max)
        print(
            f"env={env_id:03d} "
            f"origin={origins[env_id].detach().cpu().tolist()} "
            f"robot_world={robot_world[env_id].detach().cpu().tolist()} "
            f"robot_local={r_local.detach().cpu().tolist()} inside={bool(robot_inside)} "
            f"box_world={box_world[env_id].detach().cpu().tolist()} "
            f"box_local={b_local.detach().cpu().tolist()} inside={bool(box_inside)}",
            flush=True,
        )


def main():
    if not args_cli.task.startswith("ATEC-TaskD-RL-B2"):
        raise ValueError("This inspector is Task-D RL B2 only.")

    print(f"[TaskD inspect] parsing env cfg: task={args_cli.task} num_envs={args_cli.num_envs}", flush=True)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    sync_task_d_terrain_grid(env_cfg)
    env_cfg.debug_reset_positions = args_cli.debug_reset_positions
    env_cfg.debug_reset_num_envs = args_cli.debug_reset_num_envs

    print("[TaskD inspect] creating gym env", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    print("[TaskD inspect] gym env created", flush=True)
    try:
        print("[TaskD inspect] calling env.reset()", flush=True)
        env.reset()
        print("[TaskD inspect] after reset", flush=True)
        _print_pose_table(env, args_cli.terrain_x, args_cli.terrain_y, args_cli.terrain_origin_x, args_cli.terrain_origin_y)

        zero_actions = torch.zeros(
            (env.unwrapped.num_envs, env.unwrapped.action_manager.total_action_dim),
            device=env.unwrapped.device,
        )

        if args_cli.probe_local_x is not None:
            robot = env.unwrapped.scene["robot"]
            env_ids = torch.arange(env.unwrapped.num_envs, device=env.unwrapped.device)
            state = robot.data.root_state_w[env_ids].clone()
            state[:, 0] = env.unwrapped.scene.env_origins[:, 0] + float(args_cli.probe_local_x)
            robot.write_root_pose_to_sim(state[:, :7], env_ids=env_ids)
            robot.write_root_velocity_to_sim(torch.zeros_like(state[:, 7:13]), env_ids=env_ids)
            _, reward, terminated, truncated, _ = env.step(zero_actions)
            print(
                f"[TaskD inspect] probe_local_x={args_cli.probe_local_x} "
                f"reward={reward.detach().cpu().tolist()} "
                f"terminated={terminated.detach().cpu().tolist()} truncated={truncated.detach().cpu().tolist()}",
                flush=True,
            )
            _print_pose_table(env, args_cli.terrain_x, args_cli.terrain_y, args_cli.terrain_origin_x, args_cli.terrain_origin_y)

        if args_cli.steps > 0:
            for _ in range(args_cli.steps):
                env.step(zero_actions)
            print(f"[TaskD inspect] after {args_cli.steps} zero-action step(s)")
            _print_pose_table(env, args_cli.terrain_x, args_cli.terrain_y, args_cli.terrain_origin_x, args_cli.terrain_origin_y)
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
