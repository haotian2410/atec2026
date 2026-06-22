from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _as_range(value: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, tuple):
        return float(value[0]), float(value[1])
    return float(value), float(value)


def _sample_uniform(value: float | tuple[float, float], count: int, device: torch.device) -> torch.Tensor:
    low, high = _as_range(value)
    if low == high:
        return torch.full((count,), low, device=device)
    return torch.empty((count,), device=device).uniform_(low, high)


def _write_root_state(
    asset: Articulation | RigidObject,
    env,
    env_ids: torch.Tensor,
    pos: tuple[float | tuple[float, float], float | tuple[float, float], float | tuple[float, float]],
    yaw: float | tuple[float, float] = 0.0,
):
    count = len(env_ids)
    device = asset.device
    x = _sample_uniform(pos[0], count, device)
    y = _sample_uniform(pos[1], count, device)
    z = _sample_uniform(pos[2], count, device)
    yaw_t = _sample_uniform(yaw, count, device)
    root_pos = torch.stack([x, y, z], dim=-1) + env.scene.env_origins[env_ids]
    zeros = torch.zeros(count, device=device)
    root_quat = math_utils.quat_from_euler_xyz(zeros, zeros, yaw_t)
    root_vel = torch.zeros((count, 6), device=device)
    asset.write_root_pose_to_sim(torch.cat([root_pos, root_quat], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_vel, env_ids=env_ids)


def reset_task_d_stage(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    stage: str = "full",
    robot_default_z: float = 0.8,
    box_default_z: float = 0.5,
    mixed_stage_weights: dict[str, float] | None = None,
):
    """Reset Task-D robot and box for staged fixed-map curriculum training."""
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)

    robot: Articulation = env.scene["robot"]
    box: RigidObject = env.scene["box"]

    if stage == "mixed":
        if mixed_stage_weights is None:
            mixed_stage_weights = {"push": 0.35, "climb": 0.30, "drop": 0.20, "full": 0.15}
        stage_names = tuple(mixed_stage_weights.keys())
        if len(stage_names) == 0:
            raise ValueError("mixed_stage_weights must contain at least one stage.")
        if any(name not in ("push", "climb", "drop", "full") for name in stage_names):
            raise ValueError(f"Unsupported Task-D mixed reset stages: {stage_names}")
        weights = torch.tensor([float(mixed_stage_weights[name]) for name in stage_names], device=env.device)
        if torch.any(weights < 0.0) or float(weights.sum().item()) <= 0.0:
            raise ValueError("mixed_stage_weights must be non-negative and sum to a positive value.")
        sampled = torch.multinomial(weights / weights.sum(), len(env_ids), replacement=True)
        for stage_index, stage_name in enumerate(stage_names):
            selected = env_ids[sampled == stage_index]
            if len(selected) == 0:
                continue
            reset_task_d_stage(
                env,
                selected,
                stage=stage_name,
                robot_default_z=robot_default_z,
                box_default_z=box_default_z,
            )
        return

    if stage == "climb":
        robot_pos = ((-1.85, -1.75), (1.55, 1.65), robot_default_z)
        robot_yaw = (-0.08, 0.08)
        box_pos = ((-1.03, -0.97), (1.57, 1.63), box_default_z)
        box_yaw = 0.0
    elif stage == "drop":
        robot_pos = ((0.05, 0.25), (1.50, 1.70), 1.55)
        robot_yaw = (-0.08, 0.08)
        box_pos = ((-1.03, -0.97), (1.57, 1.63), box_default_z)
        box_yaw = 0.0
    elif stage == "push":
        robot_pos = ((-3.15, -2.85), (1.15, 1.45), robot_default_z)
        robot_yaw = (0.0, 0.35)
        box_pos = (-3.0, 1.6, box_default_z)
        box_yaw = 0.0
    elif stage == "full":
        robot_pos = (-3.0, 0.0, robot_default_z)
        robot_yaw = 0.0
        box_pos = (-3.0, 1.6, box_default_z)
        box_yaw = 0.0
    else:
        raise ValueError(f"Unknown Task-D reset stage: {stage}")

    _write_root_state(robot, env, env_ids, robot_pos, robot_yaw)
    _write_root_state(box, env, env_ids, box_pos, box_yaw)

    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos[env_ids],
        torch.zeros_like(robot.data.default_joint_vel[env_ids]),
        env_ids=env_ids,
    )
