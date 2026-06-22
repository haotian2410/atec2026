# Created by skywoodsz on 4/4/26.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase

from atec_rl_lab.tasks.task_d import constants as task_d_constants

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _local_root_pos(env: ManagerBasedRLEnv, asset) -> torch.Tensor:
    return asset.data.root_pos_w - env.scene.env_origins


def robot_x_greater_than(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    x_threshold: float = 5.0,
) -> torch.Tensor:
    """Terminate when robot local x is greater than threshold."""
    robot = env.scene[asset_cfg.name]
    return _local_root_pos(env, robot)[:, 0] > float(x_threshold)


def box_near_target_xy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    target_xy: tuple[float, float] = (task_d_constants.CLIMB_BOX_TARGET_X, task_d_constants.CLIMB_BOX_TARGET_Y),
    radius: float = 0.25,
) -> torch.Tensor:
    """Terminate when box xy is close enough to the fixed push target."""
    box = env.scene[asset_cfg.name]
    target = torch.tensor(target_xy, device=box.data.root_pos_w.device, dtype=box.data.root_pos_w.dtype)
    distance = torch.linalg.norm(_local_root_pos(env, box)[:, :2] - target.unsqueeze(0), dim=1)
    return distance < float(radius)

def _yaw_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Return yaw from Isaac Lab wxyz quaternions."""
    w, x, y, z = quat.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _abs_wrapped_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.abs(torch.atan2(torch.sin(angle), torch.cos(angle)))


def push_ready_for_climb(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    box_target_xy: tuple[float, float] = (task_d_constants.CLIMB_BOX_TARGET_X, task_d_constants.CLIMB_BOX_TARGET_Y),
    box_x_half_width: float = task_d_constants.CLIMB_BOX_HALF_WIDTH_X,
    box_y_half_width: float = task_d_constants.CLIMB_BOX_HALF_WIDTH_Y,
    box_max_abs_yaw: float = task_d_constants.CLIMB_BOX_YAW_TOL,
    robot_target_xy: tuple[float, float] = (task_d_constants.PRE_CLIMB_ROBOT_X, task_d_constants.PRE_CLIMB_ROBOT_Y),
    robot_x_half_width: float = task_d_constants.PRE_CLIMB_ROBOT_HALF_WIDTH_X,
    robot_y_half_width: float = task_d_constants.PRE_CLIMB_ROBOT_HALF_WIDTH_Y,
    robot_max_abs_yaw: float = task_d_constants.PRE_CLIMB_YAW_RANGE[1],
    max_projected_gravity_xy: float = 0.6,
    robot_max_lin_speed: float = 0.7,
    robot_max_ang_speed: float = 1.2,
    box_max_lin_speed: float = 0.35,
    box_max_ang_speed: float = 0.8,
) -> torch.Tensor:
    """Terminate Push only when the state is ready to continue from the Climb reset distribution."""
    robot = env.scene[robot_cfg.name]
    box = env.scene[box_cfg.name]

    box_target = torch.tensor(box_target_xy, device=box.data.root_pos_w.device, dtype=box.data.root_pos_w.dtype)
    box_delta = torch.abs(_local_root_pos(env, box)[:, :2] - box_target.unsqueeze(0))
    box_near = (box_delta[:, 0] < float(box_x_half_width)) & (box_delta[:, 1] < float(box_y_half_width))
    box_yaw_ok = _abs_wrapped_angle(_yaw_from_quat_wxyz(box.data.root_quat_w) - 0.0) < float(box_max_abs_yaw)

    robot_target = torch.tensor(robot_target_xy, device=robot.data.root_pos_w.device, dtype=robot.data.root_pos_w.dtype)
    robot_delta = torch.abs(_local_root_pos(env, robot)[:, :2] - robot_target.unsqueeze(0))
    robot_near = (robot_delta[:, 0] < float(robot_x_half_width)) & (robot_delta[:, 1] < float(robot_y_half_width))
    robot_yaw_ok = _abs_wrapped_angle(_yaw_from_quat_wxyz(robot.data.root_quat_w) - 0.0) < float(robot_max_abs_yaw)
    robot_upright = torch.linalg.norm(robot.data.projected_gravity_b[:, :2], dim=1) < float(max_projected_gravity_xy)

    robot_slow = (
        torch.linalg.norm(robot.data.root_lin_vel_w, dim=1) < float(robot_max_lin_speed)
    ) & (torch.linalg.norm(robot.data.root_ang_vel_w, dim=1) < float(robot_max_ang_speed))
    box_slow = (
        torch.linalg.norm(box.data.root_lin_vel_w, dim=1) < float(box_max_lin_speed)
    ) & (torch.linalg.norm(box.data.root_ang_vel_w, dim=1) < float(box_max_ang_speed))

    return box_near & box_yaw_ok & robot_near & robot_yaw_ok & robot_upright & robot_slow & box_slow

class JointPositionHardLimit(ManagerTermBase):
    """Terminate after controlled joints remain beyond hard safety limits for several frames."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._violation_count = torch.zeros(self._env.num_envs, dtype=torch.long, device=self._env.device)
        self._steps_since_reset = torch.zeros(self._env.num_envs, dtype=torch.long, device=self._env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self._violation_count.zero_()
            self._steps_since_reset.zero_()
        else:
            self._violation_count[env_ids] = 0
            self._steps_since_reset[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        hard_margin: float = 0.20,
        consecutive_frames: int = 8,
        grace_steps: int = 10,
        debug: bool = False,
        debug_num_envs: int = 4,
    ) -> torch.Tensor:
        robot = env.scene[asset_cfg.name]
        joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
        joint_pos = robot.data.joint_pos[:, joint_ids]
        limits = robot.data.soft_joint_pos_limits[:, joint_ids]
        lower = limits[..., 0] - float(hard_margin)
        upper = limits[..., 1] + float(hard_margin)
        raw_violations = (joint_pos < lower) | (joint_pos > upper)
        violating = torch.any(raw_violations, dim=1)

        self._steps_since_reset += 1
        in_grace = self._steps_since_reset <= max(int(grace_steps), 0)
        violating &= ~in_grace

        if debug and torch.any(violating):
            env_ids = torch.nonzero(violating, as_tuple=False).flatten()[: int(debug_num_envs)]
            joint_names = getattr(asset_cfg, "joint_names", None)
            for env_id_t in env_ids:
                env_id = int(env_id_t.item())
                bad_joint_ids = torch.nonzero(raw_violations[env_id], as_tuple=False).flatten()
                for local_joint_id_t in bad_joint_ids[:4]:
                    local_joint_id = int(local_joint_id_t.item())
                    joint_name = joint_names[local_joint_id] if joint_names is not None and local_joint_id < len(joint_names) else str(local_joint_id)
                    pos = float(joint_pos[env_id, local_joint_id].item())
                    lo = float(lower[env_id, local_joint_id].item())
                    hi = float(upper[env_id, local_joint_id].item())
                    print(
                        "[TaskD joint limit] "
                        f"env_id={env_id} joint={joint_name} pos={pos:.4f} "
                        f"allowed=[{lo:.4f}, {hi:.4f}] steps_since_reset={int(self._steps_since_reset[env_id].item())}"
                    )

        self._violation_count = torch.where(
            violating,
            self._violation_count + 1,
            torch.zeros_like(self._violation_count),
        )
        return self._violation_count >= max(int(consecutive_frames), 1)
