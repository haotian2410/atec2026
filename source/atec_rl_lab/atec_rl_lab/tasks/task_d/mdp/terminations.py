# Created by skywoodsz on 4/4/26.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase

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
    target_xy: tuple[float, float] = (2.0, 1.6),
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
    box_target_xy: tuple[float, float] = (2.0, 1.6),
    box_radius: float = 0.10,
    box_max_abs_yaw: float = 0.15,
    robot_target_xy: tuple[float, float] = (1.20, 1.60),
    robot_x_half_width: float = 0.25,
    robot_y_half_width: float = 0.25,
    robot_max_abs_yaw: float = 0.25,
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
    box_dist = torch.linalg.norm(_local_root_pos(env, box)[:, :2] - box_target.unsqueeze(0), dim=1)
    box_near = box_dist < float(box_radius)
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

    def reset(self, env_ids=None):
        if env_ids is None:
            self._violation_count.zero_()
        else:
            self._violation_count[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        hard_margin: float = 0.10,
        consecutive_frames: int = 3,
    ) -> torch.Tensor:
        robot = env.scene[asset_cfg.name]
        joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
        joint_pos = robot.data.joint_pos[:, joint_ids]
        limits = robot.data.soft_joint_pos_limits[:, joint_ids]
        lower = limits[..., 0] - float(hard_margin)
        upper = limits[..., 1] + float(hard_margin)
        violating = torch.any((joint_pos < lower) | (joint_pos > upper), dim=1)
        self._violation_count = torch.where(
            violating,
            self._violation_count + 1,
            torch.zeros_like(self._violation_count),
        )
        return self._violation_count >= max(int(consecutive_frames), 1)
