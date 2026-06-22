#!/usr/bin/env python3  # 允许直接执行该脚本，例如 ./scripts/rsl_rl/train_task_d.py。
"""Train the Task-D B2Piper box-as-step policy with RSL-RL."""  # 脚本用途：训练任务 D B2Piper 垫高上平台策略。

from __future__ import annotations  # 启用延迟类型注解，保持 Python 版本兼容。

import runpy  # 用来在当前进程中运行通用 scripts/rsl_rl/train.py。
import sys  # 用来修改命令行参数，把默认 task 和预训练路径补进去。
from pathlib import Path  # 用 Path 检查默认预训练模型是否存在、定位 train.py。

DEFAULT_TASK = "ATEC-TaskD-RL-B2Piper-v0"  # 默认训练的 Gym task id，对应 task_d/rl_env_cfg.py。
DEFAULT_PRETRAINED_CHECKPOINT = "atec_robot_model/baseline/unitree_b2_flat/policy.pt"  # 默认 walking baseline，用作弱预训练初始化。


def main():  # 脚本主函数。
    if "--task" not in sys.argv:  # 如果用户没有手动指定 --task。
        sys.argv.extend(["--task", DEFAULT_TASK])  # 自动补上任务 D RL 训练 task。
    if "--pretrained_checkpoint" not in sys.argv and Path(DEFAULT_PRETRAINED_CHECKPOINT).exists():  # 如果用户没有指定预训练，且默认 baseline 文件存在。
        sys.argv.extend(["--pretrained_checkpoint", DEFAULT_PRETRAINED_CHECKPOINT])  # 自动使用比赛目录里的 B2 flat walking policy 做部分初始化。

    train_py = Path(__file__).with_name("train.py")  # 找到同目录下通用 RSL-RL 训练脚本 scripts/rsl_rl/train.py。
    sys.path.insert(0, str(train_py.parent))  # 将 scripts/rsl_rl 加入 import 搜索路径，保证 train.py 里的 cli_args 可导入。
    runpy.run_path(str(train_py), run_name="__main__")  # 按 __main__ 方式运行通用 train.py，复用完整训练逻辑。


if __name__ == "__main__":  # 当脚本被直接运行时。
    main()  # 启动 TaskD 默认训练流程。
