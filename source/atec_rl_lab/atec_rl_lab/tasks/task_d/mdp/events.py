from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject

from atec_rl_lab.tasks.task_d import constants as task_d_constants

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
) -> tuple[torch.Tensor, torch.Tensor]:
    count = len(env_ids)
    device = asset.device
    x = _sample_uniform(pos[0], count, device)
    y = _sample_uniform(pos[1], count, device)
    z = _sample_uniform(pos[2], count, device)
    yaw_t = _sample_uniform(yaw, count, device)
    local_pos = torch.stack([x, y, z], dim=-1)
    root_pos = local_pos + env.scene.env_origins[env_ids]
    zeros = torch.zeros(count, device=device)
    root_quat = math_utils.quat_from_euler_xyz(zeros, zeros, yaw_t)
    root_vel = torch.zeros((count, 6), device=device)
    asset.write_root_pose_to_sim(torch.cat([root_pos, root_quat], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_vel, env_ids=env_ids)
    return local_pos, root_pos


def _debug_reset_positions(
    env,
    env_ids: torch.Tensor,
    stage: str,
    robot_local_pos: torch.Tensor,
    robot_world_pos: torch.Tensor,
    box_local_pos: torch.Tensor,
    box_world_pos: torch.Tensor,
    debug_num_envs: int,
):
    count = min(int(debug_num_envs), len(env_ids))
    for i in range(count):
        env_id = int(env_ids[i].item())
        env_origin = env.scene.env_origins[env_ids[i]].detach().cpu().tolist()
        robot_local = robot_local_pos[i].detach().cpu().tolist()
        robot_world = robot_world_pos[i].detach().cpu().tolist()
        box_local = box_local_pos[i].detach().cpu().tolist()
        box_world = box_world_pos[i].detach().cpu().tolist()
        print(
            "[TaskD reset] "
            f"stage={stage} env_id={env_id} "
            f"env_origin={env_origin} "
            f"robot_local={robot_local} robot_world={robot_world} "
            f"box_local={box_local} box_world={box_world}"
        )


def _write_b2_standing_joint_state(
    robot: Articulation,
    env_ids: torch.Tensor,
    debug: bool = False,
    debug_num_envs: int = 4,
):
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(robot.data.default_joint_vel[env_ids])
    joint_names = list(getattr(robot, "joint_names", ()))
    for joint_name, target_pos in task_d_constants.B2_STANDING_JOINT_POS.items():
        if joint_name in joint_names:
            joint_pos[:, joint_names.index(joint_name)] = float(target_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    if debug and len(env_ids) > 0:
        count = min(int(debug_num_envs), len(env_ids))
        print(f"[TaskD reset joints] active_joint_names={joint_names}")
        for i in range(count):
            env_id = int(env_ids[i].item())
            values = {name: float(joint_pos[i, j].item()) for j, name in enumerate(joint_names)}
            print(f"[TaskD reset joints] env_id={env_id} joint_pos={values}")
            limits = robot.data.soft_joint_pos_limits[env_ids[i]].detach().cpu()
            limit_values = {
                name: [float(limits[j, 0].item()), float(limits[j, 1].item())]
                for j, name in enumerate(joint_names)
            }
            print(f"[TaskD reset joints] env_id={env_id} soft_limits={limit_values}")


def reset_task_d_stage(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    stage: str = "full",
    robot_default_z: float = task_d_constants.B2_STANDING_ROOT_Z,
    box_default_z: float = 0.5,
    mixed_stage_weights: dict[str, float] | None = None,
    debug: bool = False,
    debug_num_envs: int = 4,
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
                debug=debug,
                debug_num_envs=debug_num_envs,
            )
        return

    # Local reset coordinates are relative to terrain origins, not terrain cell centers.
    # Task-D terrain origin is at original map coordinate (-3, 0), so fixed-map x coordinates are shifted by +3m.
    if stage == "climb":
        robot_pos = (
            (
                task_d_constants.PRE_CLIMB_ROBOT_X - task_d_constants.PRE_CLIMB_ROBOT_HALF_WIDTH_X,
                task_d_constants.PRE_CLIMB_ROBOT_X + task_d_constants.PRE_CLIMB_ROBOT_HALF_WIDTH_X,
            ),
            (
                task_d_constants.PRE_CLIMB_ROBOT_Y - task_d_constants.PRE_CLIMB_ROBOT_HALF_WIDTH_Y,
                task_d_constants.PRE_CLIMB_ROBOT_Y + task_d_constants.PRE_CLIMB_ROBOT_HALF_WIDTH_Y,
            ),
            robot_default_z,
        )
        robot_yaw = task_d_constants.PRE_CLIMB_YAW_RANGE
        box_pos = (
            (
                task_d_constants.CLIMB_BOX_TARGET_X - task_d_constants.CLIMB_BOX_HALF_WIDTH_X,
                task_d_constants.CLIMB_BOX_TARGET_X + task_d_constants.CLIMB_BOX_HALF_WIDTH_X,
            ),
            (
                task_d_constants.CLIMB_BOX_TARGET_Y - task_d_constants.CLIMB_BOX_HALF_WIDTH_Y,
                task_d_constants.CLIMB_BOX_TARGET_Y + task_d_constants.CLIMB_BOX_HALF_WIDTH_Y,
            ),
            box_default_z,
        )
        box_yaw = task_d_constants.CLIMB_BOX_YAW
    elif stage == "drop":
        robot_pos = (task_d_constants.DROP_ROBOT_X_RANGE, task_d_constants.DROP_ROBOT_Y_RANGE, task_d_constants.DROP_ROBOT_Z)
        robot_yaw = task_d_constants.DROP_ROBOT_YAW_RANGE
        box_pos = (
            (
                task_d_constants.CLIMB_BOX_TARGET_X - task_d_constants.CLIMB_BOX_HALF_WIDTH_X,
                task_d_constants.CLIMB_BOX_TARGET_X + task_d_constants.CLIMB_BOX_HALF_WIDTH_X,
            ),
            (
                task_d_constants.CLIMB_BOX_TARGET_Y - task_d_constants.CLIMB_BOX_HALF_WIDTH_Y,
                task_d_constants.CLIMB_BOX_TARGET_Y + task_d_constants.CLIMB_BOX_HALF_WIDTH_Y,
            ),
            box_default_z,
        )
        box_yaw = task_d_constants.CLIMB_BOX_YAW
    elif stage == "push":
        # Match the competition start neighborhood: robot near original (-3, 0), box near (-3, 1.6).
        robot_pos = ((-0.15, 0.15), (-0.20, 0.20), robot_default_z)
        robot_yaw = (-0.12, 0.12)
        box_pos = ((-0.05, 0.05), (1.50, 1.70), box_default_z)
        box_yaw = (-0.04, 0.04)
    elif stage == "full":
        robot_pos = (*task_d_constants.FULL_ROBOT_START, robot_default_z)
        robot_yaw = 0.0
        box_pos = (*task_d_constants.FULL_BOX_START, box_default_z)
        box_yaw = 0.0
    else:
        raise ValueError(f"Unknown Task-D reset stage: {stage}")

    robot_local_pos, robot_world_pos = _write_root_state(robot, env, env_ids, robot_pos, robot_yaw)
    box_local_pos, box_world_pos = _write_root_state(box, env, env_ids, box_pos, box_yaw)

    if debug:
        _debug_reset_positions(
            env,
            env_ids,
            stage,
            robot_local_pos,
            robot_world_pos,
            box_local_pos,
            box_world_pos,
            debug_num_envs,
        )

    _write_b2_standing_joint_state(robot, env_ids, debug=debug, debug_num_envs=debug_num_envs)
