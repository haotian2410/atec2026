#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run Task D solution until manual climb starts and print robot/box poses.")
parser.add_argument("--task", type=str, default="ATEC-TaskD-B2Piper")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=2500)
parser.add_argument("--target_sequence_step", type=int, default=19)
parser.add_argument("--progress_every", type=int, default=100)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import atec_rl_lab.tasks  # noqa: F401,E402
from atec_rl_lab.tasks.task_base.action_base import apply_safe_action_spec  # noqa: E402
from demo.solution import AlgSolution  # noqa: E402


def quat_yaw(quat: torch.Tensor) -> float:
    # Isaac uses wxyz quaternions.
    w, x, y, z = [float(v) for v in quat.detach().cpu().tolist()]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def pose_dict(env) -> dict:
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    box = unwrapped.scene["box"]
    origin = unwrapped.scene.env_origins[0]
    robot_pos_w = robot.data.root_pos_w[0]
    box_pos_w = box.data.root_pos_w[0]
    robot_quat = robot.data.root_quat_w[0]
    box_quat = box.data.root_quat_w[0]
    robot_pos_l = robot_pos_w - origin
    box_pos_l = box_pos_w - origin
    return {
        "env_origin": [round(float(v), 6) for v in origin.detach().cpu().tolist()],
        "robot_world_pos": [round(float(v), 6) for v in robot_pos_w.detach().cpu().tolist()],
        "robot_local_pos": [round(float(v), 6) for v in robot_pos_l.detach().cpu().tolist()],
        "robot_quat_wxyz": [round(float(v), 6) for v in robot_quat.detach().cpu().tolist()],
        "robot_yaw": round(quat_yaw(robot_quat), 6),
        "box_world_pos": [round(float(v), 6) for v in box_pos_w.detach().cpu().tolist()],
        "box_local_pos": [round(float(v), 6) for v in box_pos_l.detach().cpu().tolist()],
        "box_quat_wxyz": [round(float(v), 6) for v in box_quat.detach().cpu().tolist()],
        "box_yaw": round(quat_yaw(box_quat), 6),
        "robot_root_lin_vel": [round(float(v), 6) for v in robot.data.root_lin_vel_w[0].detach().cpu().tolist()],
        "box_root_lin_vel": [round(float(v), 6) for v in box.data.root_lin_vel_w[0].detach().cpu().tolist()],
        "robot_root_ang_vel": [round(float(v), 6) for v in robot.data.root_ang_vel_w[0].detach().cpu().tolist()],
        "box_root_ang_vel": [round(float(v), 6) for v in box.data.root_ang_vel_w[0].detach().cpu().tolist()],
        "robot_joint_names": list(getattr(robot, "joint_names", ())),
        "robot_joint_pos": [round(float(v), 6) for v in robot.data.joint_pos[0].detach().cpu().tolist()],
        "robot_joint_vel": [round(float(v), 6) for v in robot.data.joint_vel[0].detach().cpu().tolist()],
    }


def main():
    solution = AlgSolution()
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    action_spec = solution.get_action_spec() if hasattr(solution, "get_action_spec") else None
    env_cfg = apply_safe_action_spec(env_cfg, json.dumps(action_spec))
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        obs, _ = env.reset()
        total_episode_reward = 0.0
        last_record = None
        for step in range(args_cli.max_steps):
            if getattr(solution, "sequence_step", -1) >= args_cli.target_sequence_step:
                last_record = pose_dict(env)
                last_record["step"] = step
                last_record["sequence_step"] = int(solution.sequence_step)
                last_record["climb_frame"] = int(getattr(solution, "climb_frame", -1))
                break
            resp = solution.predicts(obs, total_episode_reward)
            if resp.get("giveup", False):
                print("[TaskD inspect] solution gave up before target sequence step", flush=True)
                break
            if args_cli.progress_every > 0 and step % args_cli.progress_every == 0:
                seq_step = getattr(solution, "sequence_step", -1)
                frame = getattr(solution, "frame", -1)
                print(f"[TaskD inspect] step={step} sequence_step={seq_step} frame={frame}", flush=True)
            action = torch.tensor(resp["action"], dtype=torch.float32, device=env.unwrapped.device).view(args_cli.num_envs, -1)
            obs, reward, terminated, truncated, info = env.step(action)
            if isinstance(reward, torch.Tensor):
                total_episode_reward += float(reward.mean().item()) / float(info.get("Step_dt", 1.0))
            if bool(torch.any(terminated).item()) or bool(torch.any(truncated).item()):
                print(f"[TaskD inspect] done before target at step={step} terminated={terminated.detach().cpu().tolist()} truncated={truncated.detach().cpu().tolist()}", flush=True)
                last_record = pose_dict(env)
                last_record["step"] = step
                last_record["sequence_step"] = int(getattr(solution, "sequence_step", -1))
                break
        if last_record is None:
            last_record = pose_dict(env)
            last_record["step"] = args_cli.max_steps
            last_record["sequence_step"] = int(getattr(solution, "sequence_step", -1))
        print("[TaskD inspect] pose before fixed/manual action:", flush=True)
        print(json.dumps(last_record, indent=2, sort_keys=True), flush=True)
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
