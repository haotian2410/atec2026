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

`scripts/rsl_rl/train_task_d.py` 如果不传 `--task`，默认训练 `ATEC-TaskD-RL-B2-Climb-v0`。默认不再自动加载 `atec_robot_model/baseline/unitree_b2_flat/policy.pt`，因为 Task D 的 lidar 观测维度不同，部分加载旧 walking policy 会导致初始动作过大、Climb 大量摔倒；需要对照实验时可显式加 `--use_default_pretrained`。

所有阶段的 checkpoint 默认写到：

```text
logs/rsl_rl/task_d_b2_box_step/<RUN_DIR>/model_<ITER>.pt
```

恢复训练时，把 `<RUN_DIR>` 替换成 `logs/rsl_rl/task_d_b2_box_step/` 下的具体 run 目录，把 `<CHECKPOINT>` 替换成类似 `model_2500.pt` 的 checkpoint 文件名。

Task D RL 地形会在 `--num_envs` 覆盖之后自动同步：`scene.env_spacing >= 12.5`，并且每个并行 env 对应一块独立 Task D terrain tile。因此切换 `--num_envs 4`、`512` 或其他值时，不需要手动改 terrain rows/cols。

当前建议先用 `--num_envs 512` 正式训练。`1024` 也可以试，但 Task D 地图和 lidar 输入比较重，512 更稳。

当前 RL 环境使用 plain Unitree B2，不使用 Piper 机械臂；动作空间是 12 个腿部关节。Task D RL task id 统一使用 `ATEC-TaskD-RL-B2-*`，不要再使用旧的 `B2Piper` RL 名称。

阶段之间按状态分布衔接：Push 的成功终止不再只是箱子到目标点，而是要求箱子姿态接近 Climb reset 的箱子姿态，机器人也要站到 Climb 的预爬入口附近并基本朝向正确；Climb 的平台侧奖励要求机器人在平台侧保持较稳定、低速状态，以便接上 Drop reset。当前固定地图接口坐标均为 env-origin local 坐标：Full robot start `(0.0, 0.0)`，Full box start `(0.0, 1.6)`，Push/Climb box target `(2.9, 1.6)`，Push/Climb pre-climb robot target `(1.9, 1.6)`，Drop robot reset `x=(3.05, 3.25), y=(1.50, 1.70)`。

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

## Stage 0: B2 稳定性检查

Stage 0 不单独增加新的 Task D curriculum task id。它用于确认 plain B2、12 维动作、预训练加载、reset 和视频录制都正常，再进入正式课程训练。

先列出当前 Task D RL 环境名：

```bash
python scripts/list_envs.py | grep "ATEC-TaskD-RL"
```

再跑 Climb 和 Push 的 1 iteration sanity：

```bash
python scripts/rsl_rl/train_task_d.py \
  --task ATEC-TaskD-RL-B2-Climb-v0 \
  --run_name sanity_climb \
  --num_envs 4 \
  --max_iterations 1 \
  --video \
  --video_length 200
```

```bash
python scripts/rsl_rl/train_task_d.py \
  --task ATEC-TaskD-RL-B2-Push-v0 \
  --run_name sanity_push \
  --num_envs 4 \
  --max_iterations 1 \
  --video \
  --video_length 200
```

如果视频里 B2 仍然出现剧烈抽动、明显翻滚或 reset 到地图外，先不要启动长训练。

## Stage 1: Push

从 walking baseline 开始训练推箱子阶段。当前 Push 成功终止要求同时满足：箱子接近 Climb 的箱子目标 `(2.9, 1.6)`，箱子 yaw 基本对齐，机器人接近 Climb 预爬入口 `(1.9, 1.6)`，机器人 yaw 基本对齐，并且机器人/箱子速度较低。


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

从最好的 Push checkpoint 开始训练爬箱子/上平台。Climb reset 中箱子已经在 Push 成功目标附近，机器人从预爬入口附近开始；平台侧奖励要求机器人在平台侧保持直立且速度较低，方便后续接 Drop。


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

从最好的 Climb checkpoint 开始训练下平台后稳定冲线。Drop reset 从平台侧/平台上稳定姿态开始，目标是下平台后继续向终点走。


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

从最好的 Drop checkpoint 开始混合 reset 训练，用于测试和强化 Push、Climb、Drop、Full 起点之间的衔接。Mixed 不是重新设计任务，只是按 reset 分布混合采样。


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

从最好的 Mixed checkpoint 开始完整端到端训练。如果 Mixed 表现不稳定，也可以直接用最好的 Drop 或 Climb checkpoint 进入 Full fine-tuning，再通过视频检查 Push->Climb->Drop 是否连续。


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
