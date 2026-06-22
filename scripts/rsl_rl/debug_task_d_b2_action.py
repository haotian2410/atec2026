#!/usr/bin/env python3
"""Debug Task-D B2 joint reset/action mapping without training."""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug Task-D B2 reset and action/joint mapping.")
parser.add_argument("--task", type=str, default="ATEC-TaskD-RL-B2-Climb-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--mode", choices=("zero_action", "random_action", "policy_action"), default="zero_action")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--load_run", type=str, default=None)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--debug_reset_positions", action="store_true", default=True)
parser.add_argument("--debug_reset_num_envs", type=int, default=4)
parser.add_argument("--full_output", action="store_true", default=False, help="Print full root/action diagnostics before and after every step.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner, DistillationRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import atec_rl_lab.tasks  # noqa: F401, E402
from atec_rl_lab.tasks.task_d.rl_env_cfg import sync_task_d_terrain_grid  # noqa: E402


def _tensor_list(tensor: torch.Tensor, env_id: int = 0, precision: int = 5):
    data = tensor[env_id].detach().cpu().flatten().tolist()
    return [round(float(v), precision) for v in data]


def _action_terms(env):
    manager = env.unwrapped.action_manager
    terms = []
    for name in getattr(manager, "active_terms", []):
        try:
            term = manager.get_term(name)
        except Exception:
            term = getattr(manager, "_terms", {}).get(name)
        terms.append((name, term))
    if not terms and hasattr(manager, "_terms"):
        terms = list(manager._terms.items())
    return terms


def _term_joint_names(term):
    for attr in ("joint_names", "_joint_names"):
        value = getattr(term, attr, None)
        if value is not None:
            return list(value)
    cfg = getattr(term, "cfg", None)
    value = getattr(cfg, "joint_names", None)
    return list(value) if value is not None else None


def _term_tensor(term, *names):
    for name in names:
        value = getattr(term, name, None)
        if torch.is_tensor(value):
            return value
    return None


def _print_static_info(env):
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    print("[TaskD B2 debug] task=", args_cli.task, flush=True)
    print("[TaskD B2 debug] num_envs=", unwrapped.num_envs, flush=True)
    print("[TaskD B2 debug] robot_asset=", type(robot).__name__, flush=True)
    print("[TaskD B2 debug] robot_cfg=", type(getattr(unwrapped.cfg.scene, "robot", None)).__name__, flush=True)
    print("[TaskD B2 debug] total_action_dim=", unwrapped.action_manager.total_action_dim, flush=True)
    print("[TaskD B2 debug] asset_joint_names=", list(getattr(robot, "joint_names", ())), flush=True)

    for name, term in _action_terms(env):
        cfg = getattr(term, "cfg", None)
        print(f"[TaskD B2 debug] action_term={name} type={type(term).__name__}")
        print(
            f"[TaskD B2 debug] action_cfg {name}: "
            f"type={type(cfg).__name__} scale={getattr(cfg, 'scale', None)} "
            f"use_default_offset={getattr(cfg, 'use_default_offset', None)} "
            f"clip={getattr(cfg, 'clip', None)} preserve_order={getattr(cfg, 'preserve_order', None)}"
        )
        print(f"[TaskD B2 debug] action_joint_names {name}=", _term_joint_names(term))
        default_offset = _term_tensor(term, "_offset", "offset", "_default_joint_pos", "default_joint_pos")
        if default_offset is not None:
            print(f"[TaskD B2 debug] action_default_or_offset {name}=", _tensor_list(default_offset))
        scale = _term_tensor(term, "_scale", "scale")
        if scale is not None:
            print(f"[TaskD B2 debug] action_scale_tensor {name}=", _tensor_list(scale))


def _joint_dict(robot, tensor):
    joint_names = list(getattr(robot, "joint_names", ()))
    return dict(zip(joint_names, _tensor_list(tensor)))


def _limit_report(robot, env_id: int = 0):
    joint_names = list(getattr(robot, "joint_names", ()))
    lower = robot.data.soft_joint_pos_limits[:, :, 0]
    upper = robot.data.soft_joint_pos_limits[:, :, 1]
    joint_pos = robot.data.joint_pos
    lower_dist = joint_pos - lower
    upper_dist = upper - joint_pos
    violating = (joint_pos[env_id] < lower[env_id]) | (joint_pos[env_id] > upper[env_id])
    violations = []
    if bool(torch.any(violating).item()):
        for idx_t in torch.nonzero(violating, as_tuple=False).flatten():
            idx = int(idx_t.item())
            pos = float(joint_pos[env_id, idx].item())
            lo = float(lower[env_id, idx].item())
            hi = float(upper[env_id, idx].item())
            amount = max(lo - pos, pos - hi, 0.0)
            violations.append((joint_names[idx], pos, lo, hi, amount))
    return round(float(torch.min(lower_dist[env_id]).item()), 5), round(float(torch.min(upper_dist[env_id]).item()), 5), violations


def _action_targets(env):
    targets = {}
    for name, term in _action_terms(env):
        for attr in ("_joint_pos_target", "joint_pos_target", "processed_actions", "_processed_actions", "_raw_actions", "raw_actions"):
            value = getattr(term, attr, None)
            if torch.is_tensor(value):
                targets[f"{name}.{attr}"] = _tensor_list(value)
    return targets


def _print_reset_state(env):
    robot = env.unwrapped.scene["robot"]
    print("[TaskD B2 debug] ===== after_reset_no_step =====", flush=True)
    print("[TaskD B2 debug] root_pos_w_env0=", _tensor_list(robot.data.root_pos_w), flush=True)
    print("[TaskD B2 debug] root_quat_w_env0=", _tensor_list(robot.data.root_quat_w), flush=True)
    print("[TaskD B2 debug] root_lin_vel_w_env0=", _tensor_list(robot.data.root_lin_vel_w), flush=True)
    print("[TaskD B2 debug] root_ang_vel_w_env0=", _tensor_list(robot.data.root_ang_vel_w), flush=True)
    print("[TaskD B2 debug] joint_pos_env0=", _joint_dict(robot, robot.data.joint_pos), flush=True)
    print("[TaskD B2 debug] joint_vel_env0=", _joint_dict(robot, robot.data.joint_vel), flush=True)
    print("[TaskD B2 debug] default_joint_pos_env0=", _joint_dict(robot, robot.data.default_joint_pos), flush=True)
    limits = robot.data.soft_joint_pos_limits[0].detach().cpu()
    joint_names = list(getattr(robot, "joint_names", ()))
    print(
        "[TaskD B2 debug] soft_joint_limits_env0=",
        {name: [round(float(limits[i, 0]), 5), round(float(limits[i, 1]), 5)] for i, name in enumerate(joint_names)},
        flush=True,
    )
    min_lower, min_upper, violations = _limit_report(robot)
    print(f"[TaskD B2 debug] reset_limit_summary min_lower={min_lower} min_upper={min_upper} violations={violations}", flush=True)
    print("[TaskD B2 debug] action_targets_after_reset=", _action_targets(env), flush=True)


def _print_step_summary(env, label: str, action: torch.Tensor, reward=None, terminated=None, truncated=None):
    robot = env.unwrapped.scene["robot"]
    min_lower, min_upper, violations = _limit_report(robot)
    print(f"[TaskD B2 debug] ===== {label} =====", flush=True)
    print("[TaskD B2 debug] action_env0=", _tensor_list(action), flush=True)
    print("[TaskD B2 debug] joint_pos_env0=", _joint_dict(robot, robot.data.joint_pos), flush=True)
    print("[TaskD B2 debug] joint_vel_env0=", _joint_dict(robot, robot.data.joint_vel), flush=True)
    print("[TaskD B2 debug] action_targets=", _action_targets(env), flush=True)
    print(f"[TaskD B2 debug] limit_summary min_lower={min_lower} min_upper={min_upper} violations={violations}", flush=True)
    if reward is not None:
        print("[TaskD B2 debug] reward=", reward.detach().cpu().tolist(), flush=True)
    if terminated is not None and truncated is not None:
        print("[TaskD B2 debug] terminated=", terminated.detach().cpu().tolist(), flush=True)
        print("[TaskD B2 debug] truncated=", truncated.detach().cpu().tolist(), flush=True)
    if args_cli.full_output:
        print("[TaskD B2 debug] root_pos_w_env0=", _tensor_list(robot.data.root_pos_w), flush=True)
        print("[TaskD B2 debug] root_lin_vel_w_env0=", _tensor_list(robot.data.root_lin_vel_w), flush=True)


def _print_terminations(env, terminated, truncated):
    print("[TaskD B2 debug] terminated=", terminated.detach().cpu().tolist())
    print("[TaskD B2 debug] truncated=", truncated.detach().cpu().tolist())
    manager = getattr(env.unwrapped, "termination_manager", None)
    if manager is None:
        return
    for name in getattr(manager, "active_terms", []):
        value = None
        try:
            term = manager.get_term(name)
            value = getattr(term, "_term_dones", None)
        except Exception:
            pass
        if value is None:
            value = getattr(manager, "_term_dones", {}).get(name) if hasattr(manager, "_term_dones") else None
        if torch.is_tensor(value):
            print(f"[TaskD B2 debug] termination_term {name}=", value.detach().cpu().tolist())
        else:
            print(f"[TaskD B2 debug] termination_term {name}=<not directly accessible>")


def _make_env():
    if not args_cli.task.startswith("ATEC-TaskD-RL-B2"):
        raise ValueError("This debugger is Task-D RL B2 only.")
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    sync_task_d_terrain_grid(env_cfg)
    env_cfg.debug_reset_positions = args_cli.debug_reset_positions
    env_cfg.debug_reset_num_envs = args_cli.debug_reset_num_envs
    return gym.make(args_cli.task, cfg=env_cfg)


def _load_policy(base_env):
    if args_cli.mode != "policy_action":
        return None, None
    if args_cli.checkpoint is None and args_cli.load_run is None:
        raise ValueError("policy_action mode requires --checkpoint or --load_run")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    checkpoint = args_cli.checkpoint or get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    wrapped = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    print("[TaskD B2 debug] loading_checkpoint=", checkpoint)
    runner.load(checkpoint)
    return runner.get_inference_policy(device=wrapped.unwrapped.device), wrapped


def main():
    env = _make_env()
    wrapped_env = None
    try:
        obs, _ = env.reset()
        _print_static_info(env)
        _print_reset_state(env)

        policy = None
        step_env = env
        if args_cli.mode == "policy_action":
            policy, wrapped_env = _load_policy(env)
            step_env = wrapped_env
            obs = step_env.get_observations()

        action_dim = env.unwrapped.action_manager.total_action_dim
        for step in range(args_cli.steps):
            if args_cli.mode == "zero_action":
                action = torch.zeros((env.unwrapped.num_envs, action_dim), device=env.unwrapped.device)
            elif args_cli.mode == "random_action":
                action = 0.05 * torch.randn((env.unwrapped.num_envs, action_dim), device=env.unwrapped.device)
            else:
                with torch.inference_mode():
                    action = policy(obs)
            print(f"[TaskD B2 debug] step={step}", flush=True)
            if args_cli.full_output:
                _print_step_summary(env, f"before_step_{step}", action=action)
            step_result = step_env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, _ = step_result
            else:
                obs, reward, terminated, _ = step_result
                truncated = torch.zeros_like(terminated, dtype=torch.bool)
            _print_step_summary(env, f"after_step_{step}", action=action, reward=reward, terminated=terminated, truncated=truncated)
            if args_cli.full_output:
                _print_terminations(env, terminated, truncated)
    finally:
        if wrapped_env is not None:
            wrapped_env.close()
        else:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
