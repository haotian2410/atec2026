# Task D 分阶段 RSL-RL 训练命令

本文档用于 Unitree B2 的 Task D 固定地图过拟合策略：

1. 把箱子推到平台侧
2. 爬上箱子
3. 借箱子爬上平台
4. 从平台落下
5. 走过终点线

不要在自动化脚本、代码审查脚本里启动长时间 Isaac Sim 训练。长训练请在已经配置好 Isaac Sim / Isaac Lab 的 Ubuntu 工作站上手动执行。

## 基础说明

从仓库根目录运行：

```bash
cd /home/lht/code/ATEC2026
```

`scripts/rsl_rl/train_task_d.py` 的默认行为保持不变：如果不传 `--task`，默认训练 `ATEC-TaskD-RL-B2-Climb-v0`；如果存在 `atec_robot_model/baseline/unitree_b2_flat/policy.pt`，会自动作为弱预训练初始化。

所有阶段的 checkpoint 默认写到：

```text
logs/rsl_rl/task_d_b2_box_step/<RUN_DIR>/model_<ITER>.pt
```

恢复训练时，把 `<RUN_DIR>` 替换成 `logs/rsl_rl/task_d_b2_box_step/` 下的具体 run 目录，把 `<CHECKPOINT>` 替换成类似 `model_2500.pt` 的 checkpoint 文件名。

Task D RL 地形会在 `--num_envs` 覆盖之后自动同步：`scene.env_spacing >= 12.5`，并且每个并行 env 对应一块独立 Task D terrain tile。因此切换 `--num_envs 4`、`512` 或其他值时，不需要手动改 terrain rows/cols。

当前建议先用 `--num_envs 512` 正式训练。`1024` 也可以试，但 Task D 地图和 lidar 输入比较重，512 更稳。

## 长训练前 sanity check

每次开始一个阶段长训练前，先看 reset 是否正常，再跑 1 个 PPO iteration。

以 Push 阶段为例：

```bash
python scripts/rsl_rl/inspect_task_d_reset.py \
  --task ATEC-TaskD-RL-B2-Push-v0 \
  --num_envs 4 \
  --steps 1 \
  --headless \
  --debug_reset_positions
```

```bash
python scripts/rsl_rl/train_task_d.py \
  --task ATEC-TaskD-RL-B2-Push-v0 \
  --run_name sanity_push \
  --num_envs 4 \
  --max_iterations 1 \
  --headless
```

训练其他阶段前，把 task id 中的 `Push` 换成 `Climb`、`Drop`、`Mixed` 或 `Full`，并相应修改 `--run_name`。

## Stage 1: Push

从 walking baseline 开始训练推箱子阶段：

```bash
python scripts/rsl_rl/train_task_d.py \
  --task ATEC-TaskD-RL-B2-Push-v0 \
  --run_name task_d_push \
  --num_envs 512 \
  --max_iterations 3000 \
  --headless
```

从已有 Push checkpoint 恢复训练：

```bash
python scripts/rsl_rl/train_task_d.py \
  --task ATEC-TaskD-RL-B2-Push-v0 \
  --run_name task_d_push_resume \
  --num_envs 512 \
  --max_iterations 3000 \
  --headless \
  --resume \
  --load_run <PUSH_RUN_DIR> \
  --checkpoint <PUSH_CHECKPOINT>
```

## Stage 2: Climb

从最好的 Push checkpoint 开始训练爬箱子/上平台：

```bash
python scripts/rsl_rl/train_task_d.py \
  --task ATEC-TaskD-RL-B2-Climb-v0 \
  --run_name task_d_climb_from_push \
  --num_envs 512 \
  --max_iterations 4000 \
  --headless \
  --resume \
  --load_run <PUSH_RUN_DIR> \
  --checkpoint <PUSH_CHECKPOINT>
```

如果只想跑默认 Climb 训练，这条仍然可用：

```bash
python scripts/rsl_rl/train_task_d.py --headless
```

## Stage 3: Drop

从最好的 Climb checkpoint 开始训练下平台后稳定冲线：

```bash
python scripts/rsl_rl/train_task_d.py \
  --task ATEC-TaskD-RL-B2-Drop-v0 \
  --run_name task_d_drop_from_climb \
  --num_envs 512 \
  --max_iterations 3000 \
  --headless \
  --resume \
  --load_run <CLIMB_RUN_DIR> \
  --checkpoint <CLIMB_CHECKPOINT>
```

## Stage 4: Mixed

从最好的 Drop checkpoint 开始混合 reset 训练，用于衔接各阶段：

```bash
python scripts/rsl_rl/train_task_d.py \
  --task ATEC-TaskD-RL-B2-Mixed-v0 \
  --run_name task_d_mixed_from_drop \
  --num_envs 512 \
  --max_iterations 3000 \
  --headless \
  --resume \
  --load_run <DROP_RUN_DIR> \
  --checkpoint <DROP_CHECKPOINT>
```

## Stage 5: Full

从最好的 Mixed checkpoint 开始完整端到端训练：

```bash
python scripts/rsl_rl/train_task_d.py \
  --task ATEC-TaskD-RL-B2-Full-v0 \
  --run_name task_d_full_from_mixed \
  --num_envs 512 \
  --max_iterations 6000 \
  --headless \
  --resume \
  --load_run <MIXED_RUN_DIR> \
  --checkpoint <MIXED_CHECKPOINT>
```

`ATEC-TaskD-RL-B2-v0` 也指向 Full 环境，但分阶段日志里建议显式使用 `ATEC-TaskD-RL-B2-Full-v0`，方便区分阶段。

## Play 和导出

`scripts/rsl_rl/play.py` 会加载 checkpoint，运行 policy playback，并导出：

```text
logs/rsl_rl/task_d_b2_box_step/<RUN_DIR>/exported/policy.pt
logs/rsl_rl/task_d_b2_box_step/<RUN_DIR>/exported/policy.onnx
```

播放并导出某个 Full checkpoint：

```bash
python scripts/rsl_rl/play.py \
  --task ATEC-TaskD-RL-B2-Full-v0 \
  --num_envs 16 \
  --load_run <FULL_RUN_DIR> \
  --checkpoint <FULL_CHECKPOINT> \
  --real-time
```

录制一段短视频并导出：

```bash
python scripts/rsl_rl/play.py \
  --task ATEC-TaskD-RL-B2-Full-v0 \
  --num_envs 16 \
  --load_run <FULL_RUN_DIR> \
  --checkpoint <FULL_CHECKPOINT> \
  --video \
  --video_length 600
```
