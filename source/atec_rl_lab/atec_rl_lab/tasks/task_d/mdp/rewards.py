from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

import isaaclab.sim as sim_utils
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _local_root_pos(env: ManagerBasedRLEnv, asset) -> torch.Tensor:
    return asset.data.root_pos_w - env.scene.env_origins


class RewardCrossX(ManagerTermBase):
    """One-time reward when robot crosses x threshold(s).
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        self._initialized = False
        self._reward_given = None

        # visual assets
        self._visual_spawned = False
        self._visual_prim_paths = []
        self._last_visual_update_step = -1

    def _init_buffers(self, num_thresholds: int = 1):
        if self._initialized:
            return

        self._reward_given = torch.zeros(
            (self._env.num_envs, num_thresholds),
            dtype=torch.bool,
            device=self._env.device,
        )
        self._initialized = True

    def _set_prim_color(self, prim_path: str, color: tuple[float, float, float]):
        """Set display color of a spawned cuboid prim."""
        try:
            import omni.usd
            from pxr import UsdGeom, Vt

            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                return

            gprim = UsdGeom.Gprim(prim)
            gprim.CreateDisplayColorAttr()
            gprim.GetDisplayColorAttr().Set(Vt.Vec3fArray([color]))
        except Exception as e:
            print(f"[RewardCrossX] Failed to set color for {prim_path}: {e}")

    def _spawn_threshold_assets_once(
        self,
        thresholds: Sequence[float],
        parent_prim_path: str = "/World/Visuals/RewardCrossX",
        line_length_y: float = 10.0,
        line_thickness_x: float = 0.02,
        line_height_z: float = 0.25,
        color_default: tuple[float, float, float] = (1.0, 0.2, 0.2),
    ):
        """Spawn one thin cuboid line for each threshold in each env.
        """
        if self._visual_spawned:
            return

        try:
            import omni.usd
            from pxr import UsdGeom

            stage = omni.usd.get_context().get_stage()
            if not stage.GetPrimAtPath(parent_prim_path).IsValid():
                UsdGeom.Xform.Define(stage, parent_prim_path)
        except Exception:
            pass

        self._visual_prim_paths = []

        for env_id in range(self._env.num_envs):
            env_paths = []
            for th_idx, th in enumerate(thresholds):
                prim_path = f"{parent_prim_path}/env_{env_id}_threshold_{th_idx}"

                cfg = sim_utils.CuboidCfg(
                    size=(line_thickness_x, line_length_y, line_height_z),
                    visual_material=None,
                    collision_props=None,
                    rigid_props=None,
                    mass_props=None,
                )

                cfg.func(
                    prim_path=prim_path,
                    cfg=cfg,
                    translation=(
                        float(th),
                        0.0,
                        float(3.0 + line_height_z * 0.5),
                    ),
                )

                self._set_prim_color(prim_path, color_default)
                env_paths.append(prim_path)

            self._visual_prim_paths.append(env_paths)

        self._visual_spawned = True

    def _update_threshold_asset_colors(
        self,
        color_default: tuple[float, float, float] = (1.0, 0.2, 0.2),
        color_triggered: tuple[float, float, float] = (0.2, 1.0, 0.2),
    ):
        """Update line color based on per-threshold reward status."""
        if not self._visual_spawned:
            return

        for env_id, env_paths in enumerate(self._visual_prim_paths):
            for th_idx, prim_path in enumerate(env_paths):
                color = color_triggered if self._reward_given[env_id, th_idx] else color_default
                self._set_prim_color(prim_path, color)

    def reset(self, env_ids=None):
        if not self._initialized:
            return

        if env_ids is None:
            self._reward_given.fill_(False)
        else:
            self._reward_given[env_ids] = False

        if self._visual_spawned:
            self._update_threshold_asset_colors()

    def _normalize_threshold_reward_params(
        self,
        thresholds: Sequence[float] | None,
        reward_values: Sequence[float] | None,
        threshold: float | Sequence[float],
        reward_value: float | Sequence[float],
        threshold_2: float | None,
        reward_value_2: float,
    ) -> tuple[list[float], list[float]]:
        """Normalize scalar/list args into float lists.
        """

        def _is_sequence(v) -> bool:
            return isinstance(v, Sequence) and not isinstance(v, (str, bytes))

        if thresholds is not None or reward_values is not None:
            if thresholds is None or reward_values is None:
                raise ValueError("thresholds and reward_values must be provided together.")
            thresholds_list = [float(x) for x in thresholds]
            reward_values_list = [float(x) for x in reward_values]
            if _is_sequence(threshold) or _is_sequence(reward_value) or threshold_2 is not None:
                raise ValueError(
                    "Do not mix thresholds/reward_values with threshold/reward_value/threshold_2."
                )
        else:
            th_is_seq = _is_sequence(threshold)
            rew_is_seq = _is_sequence(reward_value)
            if th_is_seq != rew_is_seq:
                raise ValueError(
                    "threshold and reward_value must both be scalar or both be list."
                )

            if th_is_seq:
                if threshold_2 is not None:
                    raise ValueError("threshold_2 cannot be used when threshold is a list.")
                thresholds_list = [float(x) for x in threshold]
                reward_values_list = [float(x) for x in reward_value]
            else:
                thresholds_list = [float(threshold)]
                reward_values_list = [float(reward_value)]
                if threshold_2 is not None:
                    thresholds_list.append(float(threshold_2))
                    reward_values_list.append(float(reward_value_2))

        if len(thresholds_list) == 0:
            raise ValueError("At least one threshold is required.")
        if len(thresholds_list) != len(reward_values_list):
            raise ValueError(
                "thresholds and reward_values must have same length, "
                f"got {len(thresholds_list)} and {len(reward_values_list)}."
            )
        for i in range(1, len(thresholds_list)):
            if thresholds_list[i] <= thresholds_list[i - 1]:
                raise ValueError("thresholds must be strictly increasing.")

        return thresholds_list, reward_values_list

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg,
        threshold: float | Sequence[float] = 0.6,
        reward_value: float | Sequence[float] = 24.0,
        threshold_2: float | None = None,
        reward_value_2: float = 0.0,
        thresholds: Sequence[float] | None = None,
        reward_values: Sequence[float] | None = None,
        debug: bool = False,
        visual_assets: bool = False,
        visual_update_interval: int = 10,
        parent_prim_path: str = "/World/Visuals/RewardCrossX",
        line_length_y: float = 10.0,
        line_thickness_x: float = 0.02,
        line_height_z: float = 0.5,
    ) -> torch.Tensor:
        thresholds, reward_values = self._normalize_threshold_reward_params(
            thresholds=thresholds,
            reward_values=reward_values,
            threshold=threshold,
            reward_value=reward_value,
            threshold_2=threshold_2,
            reward_value_2=reward_value_2,
        )

        num_thresholds = len(thresholds)

        if not self._initialized:
            self._init_buffers(num_thresholds=num_thresholds)
        elif self._reward_given.shape[1] != num_thresholds:
            raise ValueError(
                "RewardCrossX threshold count changed after initialization. "
                f"Expected {self._reward_given.shape[1]}, got {num_thresholds}."
            )

        if visual_assets and not self._visual_spawned:
            self._spawn_threshold_assets_once(
                thresholds=thresholds,
                parent_prim_path=parent_prim_path,
                line_length_y=line_length_y,
                line_thickness_x=line_thickness_x,
                line_height_z=line_height_z,
            )
        elif visual_assets and self._visual_spawned and self._visual_prim_paths:
            if len(self._visual_prim_paths[0]) != num_thresholds:
                raise ValueError(
                    "RewardCrossX visual threshold count changed after spawn. "
                    f"Expected {len(self._visual_prim_paths[0])}, got {num_thresholds}."
                )

        robot = env.scene[asset_cfg.name]

        root_pos_x = _local_root_pos(env, robot)[:, 0]
        thresholds_t = torch.tensor(thresholds, device=root_pos_x.device, dtype=root_pos_x.dtype)
        reward_values_t = torch.tensor(reward_values, device=root_pos_x.device, dtype=root_pos_x.dtype)

        crossed = root_pos_x.unsqueeze(1) > thresholds_t.unsqueeze(0)
        trigger = crossed & (~self._reward_given)
        self._reward_given |= crossed

        reward = (trigger.float() * reward_values_t.unsqueeze(0)).sum(dim=1)

        if debug:
            print(
                f"[RewardCrossX] x={root_pos_x[0].item():.3f}, "
                f"crossed={crossed[0].tolist()}, "
                f"trigger={trigger[0].tolist()}, "
                f"reward={reward[0].item():.3f}"
            )

        if visual_assets and self._visual_spawned:
            step = getattr(env, "common_step_counter", 0)
            if step != self._last_visual_update_step and step % visual_update_interval == 0:
                self._update_threshold_asset_colors()
                self._last_visual_update_step = step

        return reward


class RewardBoxXInRange(ManagerTermBase):
    """Give reward when the target box x-position is within one or more x-ranges."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._reward_given = torch.zeros(self._env.num_envs, dtype=torch.bool, device=self._env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self._reward_given.fill_(False)
        else:
            self._reward_given[env_ids] = False

    def _normalize_ranges(
        self,
        x_min: float | Sequence[float],
        x_max: float | Sequence[float],
    ) -> tuple[list[float], list[float]]:
        is_min_seq = isinstance(x_min, Sequence) and not isinstance(x_min, (str, bytes))
        is_max_seq = isinstance(x_max, Sequence) and not isinstance(x_max, (str, bytes))

        if is_min_seq != is_max_seq:
            raise ValueError("x_min and x_max must both be scalar or both be sequences.")

        if is_min_seq:
            x_min_list = [float(v) for v in x_min]
            x_max_list = [float(v) for v in x_max]
            if len(x_min_list) == 0:
                raise ValueError("At least one x-range is required.")
            if len(x_min_list) != len(x_max_list):
                raise ValueError("x_min and x_max sequences must have the same length.")
        else:
            x_min_list = [float(x_min)]
            x_max_list = [float(x_max)]

        for mn, mx in zip(x_min_list, x_max_list):
            if mn > mx:
                raise ValueError(f"Invalid x-range: x_min ({mn}) must be <= x_max ({mx}).")

        return x_min_list, x_max_list

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg,
        x_min: float | Sequence[float] = -1.0,
        x_max: float | Sequence[float] = 1.0,
        reward_value: float = 14.0,
        one_time: bool = True,
        debug: bool = False,
        y_min: float | None = None,
        y_max: float | None = None,
    ) -> torch.Tensor:
        box = env.scene[asset_cfg.name]
        box_local_pos = _local_root_pos(env, box)
        box_x = box_local_pos[:, 0]
        box_y = box_local_pos[:, 1]

        x_min_list, x_max_list = self._normalize_ranges(x_min=x_min, x_max=x_max)

        in_any_range = torch.zeros_like(box_x, dtype=torch.bool)
        for mn, mx in zip(x_min_list, x_max_list):
            in_any_range |= (box_x >= mn) & (box_x <= mx)
        if y_min is not None:
            in_any_range &= box_y >= float(y_min)
        if y_max is not None:
            in_any_range &= box_y <= float(y_max)

        if one_time:
            trigger = in_any_range & (~self._reward_given)
            self._reward_given |= in_any_range
            reward = trigger.to(box_x.dtype) * float(reward_value)
        else:
            reward = in_any_range.to(box_x.dtype) * float(reward_value)

        if debug:
            print(
                f"[RewardBoxXInRange] box_x={box_x[0].item():.3f}, "
                f"in_any_range={bool(in_any_range[0].item())}, "
                f"ranges={list(zip(x_min_list, x_max_list))}, "
                f"reward={reward[0].item():.3f}"
            )

        return reward

def joint_pos_near_soft_limits(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soft_margin: float = 0.20,
) -> torch.Tensor:
    """Penalize controlled joints that approach their soft position limits."""
    robot = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
    joint_pos = robot.data.joint_pos[:, joint_ids]
    limits = robot.data.soft_joint_pos_limits[:, joint_ids]
    lower = limits[..., 0] + float(soft_margin)
    upper = limits[..., 1] - float(soft_margin)
    lower_violation = torch.clamp(lower - joint_pos, min=0.0)
    upper_violation = torch.clamp(joint_pos - upper, min=0.0)
    return torch.sum(torch.square(lower_violation) + torch.square(upper_violation), dim=1)


class RobotForwardProgress(ManagerTermBase):  # 机器人 x 方向“增量进度”奖励，只在本步比上步更靠近终点时给分。
    def __init__(self, cfg, env):  # 初始化奖励项，IsaacLab 会在 reward manager 创建时调用。
        super().__init__(cfg, env)  # 初始化 ManagerTermBase，获得 num_envs、device 等基础字段。
        self._prev_progress = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self._env.device)  # 记录上一帧归一化进度。

    def reset(self, env_ids=None):  # episode reset 时同步上一帧进度，避免 reset 后第一步凭当前位置刷奖励。
        if env_ids is None:  # 如果没有指定 env_ids，则重置全部环境。
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)  # 构造全部环境 id。
        robot = self._env.scene["robot"]  # 读取 robot，用当前位置初始化 prev_progress。
        params = self.cfg.params or {}
        start_x = float(params.get("start_x", 0.0))
        end_x = float(params.get("end_x", 6.5))
        x = _local_root_pos(self._env, robot)[env_ids, 0]  # 读取 reset 后 robot local x。
        self._prev_progress[env_ids] = torch.clamp((x - start_x) / max(end_x - start_x, 1.0e-6), 0.0, 1.0)  # 写入当前进度。

    def __call__(  # 每个仿真 step 计算一次增量奖励。
        self,
        env: ManagerBasedRLEnv,  # 当前 ManagerBased RL 环境。
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # 指定读取 robot articulation。
        start_x: float = 0.0,  # robot 初始 local x 附近。
        end_x: float = 6.5,  # 任务 D 终点 local x。
    ) -> torch.Tensor:  # 返回每个并行环境一个 reward。
        robot = env.scene[asset_cfg.name]  # 从场景中取出 robot。
        x = _local_root_pos(env, robot)[:, 0]  # 读取 robot 当前 x。
        progress = torch.clamp((x - float(start_x)) / max(float(end_x - start_x), 1.0e-6), 0.0, 1.0)  # 当前归一化进度。
        reward = torch.clamp(progress - self._prev_progress, min=0.0)  # 只奖励正向增量，停住或后退不给分。
        self._prev_progress[:] = progress  # 更新上一帧进度，供下一步计算增量。
        return reward  # 返回增量奖励。


class BoxForwardProgress(ManagerTermBase):  # 箱子 x 方向“增量进度”奖励，只在箱子被向垫高目标推进时给分。
    def __init__(self, cfg, env):  # 初始化奖励项。
        super().__init__(cfg, env)  # 初始化 ManagerTermBase。
        self._prev_progress = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self._env.device)  # 记录上一帧箱子进度。

    def reset(self, env_ids=None):  # episode reset 时同步箱子上一帧进度。
        if env_ids is None:  # 如果没有指定 env_ids。
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)  # 使用全部环境 id。
        box = self._env.scene["box"]  # 读取箱子对象。
        params = self.cfg.params or {}
        start_x = float(params.get("start_x", 0.0))
        target_x = float(params.get("target_x", 2.0))
        x = _local_root_pos(self._env, box)[env_ids, 0]  # 读取 reset 后箱子 local x。
        self._prev_progress[env_ids] = torch.clamp((x - start_x) / max(target_x - start_x, 1.0e-6), 0.0, 1.0)  # 写入当前箱子进度。

    def __call__(  # 每步计算箱子向目标 x 推进的增量。
        self,
        env: ManagerBasedRLEnv,  # 当前环境。
        asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),  # 指定读取 box。
        start_x: float = 0.0,  # 箱子初始 local x。
        target_x: float = 2.0,  # 箱子垫高目标 local x。
    ) -> torch.Tensor:  # 返回每个并行环境一个 reward。
        box = env.scene[asset_cfg.name]  # 从场景中取出箱子。
        x = _local_root_pos(env, box)[:, 0]  # 读取箱子当前 x。
        progress = torch.clamp((x - float(start_x)) / max(float(target_x - start_x), 1.0e-6), 0.0, 1.0)  # 当前箱子进度。
        reward = torch.clamp(progress - self._prev_progress, min=0.0)  # 只奖励本步新增进度，停住不刷分。
        self._prev_progress[:] = progress  # 更新上一帧箱子进度。
        return reward  # 返回增量奖励。


def box_to_step_target(  # 箱子接近指定垫高位置的连续奖励；权重较低时可作为形状引导。
    env: ManagerBasedRLEnv,  # 当前 ManagerBased RL 环境。
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),  # 指定读取 box。
    target_xy: tuple[float, float] = (2.0, 1.6),  # 指定箱子垫高目标位置。
    std: float = 0.8,  # 指数奖励的距离尺度。
) -> torch.Tensor:  # 返回每个并行环境一个 reward tensor。
    box = env.scene[asset_cfg.name]  # 从场景中取出箱子。
    target = torch.tensor(target_xy, device=box.data.root_pos_w.device, dtype=box.data.root_pos_w.dtype)  # 目标 xy tensor。
    error = torch.sum(torch.square(_local_root_pos(env, box)[:, :2] - target.unsqueeze(0)), dim=1)  # 箱子到目标的 xy 平方距离。
    return torch.exp(-error / (float(std) ** 2))  # 距离越近奖励越接近 1。


class BoxToStepTargetProgress(ManagerTermBase):  # 箱子靠近垫高目标的增量奖励，避免到位后每步刷分。
    def __init__(self, cfg, env):  # 初始化奖励项。
        super().__init__(cfg, env)  # 初始化 ManagerTermBase。
        self._prev_distance = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self._env.device)  # 记录上一帧箱子到目标距离。

    def reset(self, env_ids=None):  # episode reset 时同步当前距离，避免 reset 后凭初始位置得分。
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        box = self._env.scene["box"]
        target_xy = self.cfg.params.get("target_xy", (2.0, 1.6)) if self.cfg.params is not None else (2.0, 1.6)
        target = torch.tensor(target_xy, device=box.data.root_pos_w.device, dtype=box.data.root_pos_w.dtype)
        self._prev_distance[env_ids] = torch.linalg.norm(_local_root_pos(self._env, box)[env_ids, :2] - target.unsqueeze(0), dim=1)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
        target_xy: tuple[float, float] = (2.0, 1.6),
        std: float = 0.8,
    ) -> torch.Tensor:
        box = env.scene[asset_cfg.name]
        target = torch.tensor(target_xy, device=box.data.root_pos_w.device, dtype=box.data.root_pos_w.dtype)
        distance = torch.linalg.norm(_local_root_pos(env, box)[:, :2] - target.unsqueeze(0), dim=1)
        improvement = torch.clamp(self._prev_distance - distance, min=0.0)
        self._prev_distance[:] = distance
        return improvement / max(float(std), 1.0e-6)


def robot_near_box(  # 机器人靠近箱子的连续奖励，降低早期探索难度。
    env: ManagerBasedRLEnv,  # 当前环境。
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # robot 实体配置。
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),  # box 实体配置。
    std: float = 1.2,  # 距离尺度。
) -> torch.Tensor:  # 返回 reward tensor。
    robot = env.scene[robot_cfg.name]  # 读取 robot。
    box = env.scene[box_cfg.name]  # 读取 box。
    error = torch.sum(torch.square(_local_root_pos(env, robot)[:, :2] - _local_root_pos(env, box)[:, :2]), dim=1)  # robot-box xy 距离平方。
    return torch.exp(-error / (float(std) ** 2))  # 指数接近奖励。


class RobotOnBoxOnce(ManagerTermBase):  # 一次性“机器人上箱子”奖励，并且要求箱子已接近垫高目标位置。
    def __init__(self, cfg, env):  # 初始化奖励项。
        super().__init__(cfg, env)  # 初始化 ManagerTermBase。
        self._reward_given = torch.zeros(self._env.num_envs, dtype=torch.bool, device=self._env.device)  # 记录每个 env 是否已经给过上箱子奖励。

    def reset(self, env_ids=None):  # episode reset 时清空一次性奖励状态。
        if env_ids is None:  # 如果没有指定 env_ids。
            self._reward_given.fill_(False)  # 清空全部环境状态。
        else:  # 只 reset 部分环境。
            self._reward_given[env_ids] = False  # 清空指定环境状态。

    def __call__(  # 每步判断是否首次满足“箱子到位 + 机器人上箱子”。
        self,
        env: ManagerBasedRLEnv,  # 当前环境。
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # robot 实体。
        box_cfg: SceneEntityCfg = SceneEntityCfg("box"),  # box 实体。
        target_xy: tuple[float, float] = (2.0, 1.6),  # 箱子垫高目标位置。
        box_ready_radius: float = 0.25,  # 箱子离目标多近才允许给上箱子奖励。
        xy_half_extents: tuple[float, float] = (0.55, 0.65),  # robot base 投影在箱子附近的 xy 判定范围。
        min_height_above_box: float = 0.45,  # robot base 必须明显高于箱子中心，避免贴箱子侧面刷分。
    ) -> torch.Tensor:  # 返回一次性 reward tensor。
        robot = env.scene[robot_cfg.name]  # 读取 robot。
        box = env.scene[box_cfg.name]  # 读取 box。
        target = torch.tensor(target_xy, device=box.data.root_pos_w.device, dtype=box.data.root_pos_w.dtype)  # 目标 xy tensor。
        box_dist = torch.linalg.norm(_local_root_pos(env, box)[:, :2] - target.unsqueeze(0), dim=1)  # 箱子到垫高目标的 xy 距离。
        box_ready = box_dist < float(box_ready_radius)  # 只有箱子接近目标位置后，才允许上箱子奖励。
        delta_xy = torch.abs(_local_root_pos(env, robot)[:, :2] - _local_root_pos(env, box)[:, :2])  # robot base 与箱子中心 xy 距离。
        within_xy = (delta_xy[:, 0] < float(xy_half_extents[0])) & (delta_xy[:, 1] < float(xy_half_extents[1]))  # 判断 robot 是否在箱子投影附近。
        above = robot.data.root_pos_w[:, 2] > box.data.root_pos_w[:, 2] + float(min_height_above_box)  # 判断 robot 是否高于箱子。
        condition = box_ready & within_xy & above  # 上箱子奖励的完整条件。
        trigger = condition & (~self._reward_given)  # 只在第一次满足条件时触发。
        self._reward_given |= condition  # 一旦满足过条件，就标记已给奖励。
        return trigger.to(robot.data.root_pos_w.dtype)  # bool 转 float，满足首次条件给 1。


class RobotOnPlatformSideOnce(ManagerTermBase):  # 一次性“机器人到达平台侧/高台区域”奖励。
    def __init__(self, cfg, env):  # 初始化奖励项。
        super().__init__(cfg, env)  # 初始化 ManagerTermBase。
        self._reward_given = torch.zeros(self._env.num_envs, dtype=torch.bool, device=self._env.device)  # 记录是否已经给过平台奖励。

    def reset(self, env_ids=None):  # episode reset 时清空一次性奖励状态。
        if env_ids is None:  # 如果 reset 全部环境。
            self._reward_given.fill_(False)  # 清空全部状态。
        else:  # 如果 reset 部分环境。
            self._reward_given[env_ids] = False  # 清空指定 env 状态。

    def __call__(  # 每步判断是否首次到达平台侧。
        self,
        env: ManagerBasedRLEnv,  # 当前环境。
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # robot 实体。
        min_x: float = 2.8,  # x 方向必须已经越过坑/平台前缘附近。
        min_z: float = 1.35,  # base 高度阈值；平台 1.0-1.2m，B2 站立 base 应明显高于 1.25m。
        target_y: float = 1.6,  # 固定垫高策略的右侧路线 y 位置。
        y_half_width: float = 1.4,  # y 容差，避免策略在错误侧或掉坑时刷平台奖励。
        max_projected_gravity_xy: float = 0.8,  # 姿态稳定性阈值，避免翻倒状态也拿平台奖励。
        target_xy: tuple[float, float] = (2.0, 1.6),  # 箱子垫高目标位置。
        box_ready_radius: float = 0.25,  # 箱子离目标足够近后才允许给平台侧奖励。
        check_box_ready: bool = True,  # 默认启用 box-ready gate；可显式关闭以兼容特殊实验。
        box_cfg: SceneEntityCfg = SceneEntityCfg("box"),  # box 实体，用于确认箱子已经在垫高目标附近。
        max_lin_speed: float = 1.0,  # 平台侧奖励要求低速稳定，使 Climb 终态更接近 Drop reset。
        max_ang_speed: float = 1.5,  # 限制角速度，避免弹飞/翻滚瞬间触发平台奖励。
    ) -> torch.Tensor:  # 返回一次性 reward tensor。
        robot = env.scene[asset_cfg.name]  # 读取 robot。
        root_pos = _local_root_pos(env, robot)  # 读取 robot root 在各自 terrain origin 下的局部坐标。
        in_x = root_pos[:, 0] > float(min_x)  # x 必须到达平台侧。
        in_y = torch.abs(root_pos[:, 1] - float(target_y)) < float(y_half_width)  # y 必须在固定右侧路线附近。
        high_enough = root_pos[:, 2] > float(min_z)  # base 高度必须足够高，避免普通站立或坑内状态刷分。
        gravity_xy = torch.linalg.norm(robot.data.projected_gravity_b[:, :2], dim=1)  # projected gravity xy 越小表示越接近直立。
        stable = gravity_xy < float(max_projected_gravity_xy)  # 姿态不能过度倾斜/翻倒。
        slow = (torch.linalg.norm(robot.data.root_lin_vel_w, dim=1) < float(max_lin_speed)) & (
            torch.linalg.norm(robot.data.root_ang_vel_w, dim=1) < float(max_ang_speed)
        )  # 速度不能太大，避免跳跃/弹飞状态接到 Drop reset。
        condition = in_x & in_y & high_enough & stable & slow  # 平台奖励完整条件。
        if check_box_ready:
            box = env.scene[box_cfg.name]  # 读取 box。
            target = torch.tensor(target_xy, device=box.data.root_pos_w.device, dtype=box.data.root_pos_w.dtype)  # 目标 xy tensor。
            box_dist = torch.linalg.norm(_local_root_pos(env, box)[:, :2] - target.unsqueeze(0), dim=1)  # 箱子到目标的 xy 距离。
            condition &= box_dist < float(box_ready_radius)  # 箱子未到位时不允许刷平台侧奖励。
        trigger = condition & (~self._reward_given)  # 只在第一次满足时触发。
        self._reward_given |= condition  # 标记已给奖励。
        return trigger.to(robot.data.root_pos_w.dtype)  # bool 转 float。
