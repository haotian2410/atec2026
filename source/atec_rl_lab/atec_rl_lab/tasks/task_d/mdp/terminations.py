# Created by skywoodsz on 4/4/26.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

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
