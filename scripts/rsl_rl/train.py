
"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")  # 训练任务名，例如 ATEC-TaskD-RL-B2-v0。
parser.add_argument(  # RL agent 配置入口名，默认读取 gym.register 里的 rsl_rl_cfg_entry_point。
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")  # 随机种子；不传则使用 agent cfg 默认种子。
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")  # 覆盖 PPO 最大训练轮数。
parser.add_argument(  # 可选预训练权重路径；这里只做部分初始化，不是 resume。
    "--pretrained_checkpoint",  # 参数名，例如 --pretrained_checkpoint atec_robot_model/baseline/unitree_b2_flat/policy.pt。
    type=str,  # 路径用字符串表示。
    default=None,  # 默认不启用；train_task_d.py 会自动补默认 baseline。
    help="Optional checkpoint or TorchScript policy used only for partial actor-critic initialization.",  # 说明该参数只用于 actor-critic 弱初始化。
)
parser.add_argument(  # 多 GPU/多节点训练开关，保留通用 RSL-RL 能力。
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")  # 导出 ManagerBased env IO 描述，默认关闭。
parser.add_argument(  # Ray 集成时使用的进程 id，普通本地训练不用管。
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import logging
import os
import time
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import atec_rl_lab.train  # noqa: F401  # isort: skip
import atec_rl_lab.tasks  # noqa: F401  # isort: skip


def _sync_task_d_terrain_if_needed(env_cfg):
    if not hasattr(env_cfg, "task_d_stage"):
        return
    from atec_rl_lab.tasks.task_d.rl_env_cfg import sync_task_d_terrain_grid

    sync_task_d_terrain_grid(env_cfg)


# import logger
logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _load_partial_pretrained_policy(runner, checkpoint_path: str):  # 从已有 walking policy/checkpoint 对当前 actor-critic 做“能匹配就加载”的弱初始化。
    """Partially initialize actor-critic weights from an existing walking policy.

    This is intentionally not a resume path: optimizer state, iteration counters, and
    mismatched input/output layers are ignored. It is useful when a locomotion policy
    should only provide a mild initialization for a different task observation/action
    space.
    """
    if checkpoint_path is None:  # 没有传预训练路径时直接返回。
        return  # 不改变当前随机初始化模型。
    checkpoint_path = os.path.abspath(os.path.expanduser(checkpoint_path))  # 展开 ~ 并转为绝对路径，避免工作目录变化导致找不到文件。
    if not os.path.exists(checkpoint_path):  # 检查预训练文件是否存在。
        raise FileNotFoundError(f"Pretrained checkpoint does not exist: {checkpoint_path}")  # 文件不存在时明确报错。

    target_module = None  # 当前 runner 里的 actor-critic 网络模块，后面兼容式查找。
    candidate_roots = [runner, getattr(runner, "alg", None)]  # RSL-RL 版本不同，网络可能挂在 runner 或 runner.alg 下。
    candidate_attrs = (  # 常见 actor-critic/policy 属性名列表。
        "actor_critic",  # 旧版或部分 runner 常用属性名。
        "policy",  # RSL-RL 3.x 里 PPO 可能把 actor-critic 叫 policy。
        "actor_critic_policy",  # 兼容可能的封装命名。
        "student_actor_critic",  # 兼容蒸馏/student runner 命名。
        "teacher_actor_critic",  # 兼容 teacher runner 命名。
    )
    for root in candidate_roots:  # 遍历 runner 和 runner.alg。
        if root is None:  # 某些 runner 可能没有 alg。
            continue  # 跳过空对象。
        for attr in candidate_attrs:  # 遍历候选属性名。
            module = getattr(root, attr, None)  # 尝试读取该属性。
            if module is not None and hasattr(module, "state_dict"):  # 找到像 PyTorch module 的对象。
                target_module = module  # 记录当前训练网络。
                break  # 停止属性名搜索。
        if target_module is not None:  # 已经找到网络模块。
            break  # 停止 root 搜索。
    if target_module is None:  # 没找到 actor-critic 模块。
        alg_attrs = sorted(k for k in dir(getattr(runner, "alg", object())) if not k.startswith("_"))  # 收集 alg 属性预览，方便适配新版本。
        runner_attrs = sorted(k for k in dir(runner) if not k.startswith("_"))  # 收集 runner 属性预览。
        raise AttributeError(  # 抛出带属性预览的错误，方便继续兼容 RSL-RL 版本。
            "Could not find actor-critic module for partial pretrained initialization. "
            f"runner attrs preview={runner_attrs[:30]}, alg attrs preview={alg_attrs[:30]}"
        )
    target_state = target_module.state_dict()  # 读取当前 TaskD actor-critic 的参数字典。

    source_state = None  # 预训练模型的参数字典。
    try:  # 优先按普通 torch checkpoint 读取。
        checkpoint = torch.load(checkpoint_path, map_location="cpu")  # 加载到 CPU，避免初始化阶段占用额外 GPU 显存。
        if isinstance(checkpoint, dict):  # RSL-RL checkpoint 通常是 dict。
            for key in ("model_state_dict", "state_dict", "actor_critic_state_dict", "policy_state_dict"):  # 常见 state_dict key。
                if key in checkpoint and isinstance(checkpoint[key], dict):  # 找到嵌套参数字典。
                    source_state = checkpoint[key]  # 使用该嵌套参数字典。
                    break  # 停止 key 搜索。
            if source_state is None and all(torch.is_tensor(v) for v in checkpoint.values()):  # 如果 dict 本身就是 tensor 参数表。
                source_state = checkpoint  # 直接把 checkpoint 当 state_dict。
    except Exception as exc:  # torch.load 失败可能是 TorchScript policy.pt。
        print(f"[INFO] torch.load could not read pretrained checkpoint as a state dict: {exc}")  # 打印信息，随后尝试 jit.load。

    if source_state is None:  # 普通 checkpoint 没解析出来。
        try:  # 尝试按 TorchScript 读取导出的 policy.pt。
            scripted = torch.jit.load(checkpoint_path, map_location="cpu")  # 加载 TorchScript policy。
            source_state = scripted.state_dict()  # 取 TorchScript 模块的 state_dict。
        except Exception as exc:  # 两种格式都读不了。
            raise RuntimeError(  # 明确提示预训练文件格式无法解析。
                f"Could not load pretrained checkpoint as either torch checkpoint or TorchScript: {checkpoint_path}"
            ) from exc

    def candidate_keys(src_key: str):  # 将预训练参数名映射成若干可能的当前网络参数名。
        keys = [src_key]  # 原名优先匹配。
        for prefix in ("module.", "alg.actor_critic.", "actor_critic."):  # 去掉常见封装前缀。
            if src_key.startswith(prefix):  # 如果源参数名有该前缀。
                keys.append(src_key[len(prefix):])  # 加入去前缀后的候选名。
        keys.append("actor." + src_key)  # 兼容导出 policy 只保存 actor 子模块时的命名。
        return keys  # 返回候选参数名列表。

    updates = {}  # 将要加载到当前网络的参数。
    skipped = []  # 记录跳过参数及原因，方便用户判断到底加载了多少。
    for src_key, src_value in source_state.items():  # 遍历预训练参数。
        if not torch.is_tensor(src_value):  # 非 tensor 项不能作为权重加载。
            continue  # 跳过元数据。
        matched_key = None  # 当前网络中匹配到的参数名。
        for key in candidate_keys(src_key):  # 尝试所有候选参数名。
            if key in target_state:  # 当前网络存在该参数。
                matched_key = key  # 记录匹配名。
                break  # 停止候选搜索。
        if matched_key is None:  # 当前网络没有对应参数名。
            skipped.append((src_key, "missing"))  # 记录因缺失跳过。
            continue  # 不加载该参数。
        if tuple(src_value.shape) != tuple(target_state[matched_key].shape):  # 形状不一致，例如 lidar 输入导致第一层维度不同。
            skipped.append((src_key, f"shape {tuple(src_value.shape)} -> {tuple(target_state[matched_key].shape)}"))  # 记录形状不匹配。
            continue  # 保持当前随机初始化，不强行拷贝。
        updates[matched_key] = src_value.to(dtype=target_state[matched_key].dtype)  # 形状一致则转换 dtype 后准备加载。

    target_state.update(updates)  # 把匹配成功的预训练参数覆盖到当前参数表。
    target_module.load_state_dict(target_state, strict=True)  # 严格加载完整当前参数表；未匹配项仍保留随机初始化值。
    print(  # 打印加载统计。
        f"[INFO] Partially initialized actor-critic from {checkpoint_path}: "
        f"loaded {len(updates)} tensors, skipped {len(skipped)} tensors."
    )
    if skipped:  # 如果有跳过项。
        preview = ", ".join(f"{name} ({reason})" for name, reason in skipped[:8])  # 只展示前 8 个，避免日志太长。
        print(f"[INFO] Pretrained skipped preview: {preview}")  # 打印跳过项预览。


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    _sync_task_d_terrain_if_needed(env_cfg)
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not
    # change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    if args_cli.pretrained_checkpoint and not agent_cfg.resume:  # 仅在非 resume 训练时使用 walking baseline 做弱初始化。
        _load_partial_pretrained_policy(runner, args_cli.pretrained_checkpoint)  # 只加载形状匹配的权重，输入/输出不匹配层保持随机初始化。

    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    init_at_random_ep_len = not hasattr(env_cfg, "task_d_stage")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=init_at_random_ep_len)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
