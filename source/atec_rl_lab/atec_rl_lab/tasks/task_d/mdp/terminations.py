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
