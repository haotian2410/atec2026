import gymnasium as gym

from .terrain import TASK_D_TERRAIN_CFG
from .env_cfg import TaskDEnvCfg, TaskDEnvB2Cfg, TaskDEnvTron2ALeggedCfg, TaskDEnvTron2AWheelCfg
from .rl_env_cfg import TaskDRLEnvB2Cfg, TaskDRLEnvB2ClimbCfg, TaskDRLEnvB2DropCfg, TaskDRLEnvB2PushCfg, TaskDRLEnvB2MixedCfg, TaskDRLEnvB2FullCfg
from . import agents


gym.register(
    id = "ATEC-TaskD-G1",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskDEnvG1Cfg"
    },
)

gym.register(
    id = "ATEC-TaskD-Tron1Piper",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskDEnvTron1Cfg"
    },
)

gym.register(
    id = "ATEC-TaskD-Tron2ALegged",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskDEnvTron2ALeggedCfg"
    },
)

gym.register(
    id = "ATEC-TaskD-Tron2AWheel",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskDEnvTron2AWheelCfg"
    },
)

gym.register(
    id = "ATEC-TaskD-B2Piper",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskDEnvB2Cfg"
    },
)

gym.register(
    id = "ATEC-TaskD-B2wPiper",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskDEnvB2WCfg"
    },
)



gym.register(
    id="ATEC-TaskD-RL-B2-Climb-v0",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_env_cfg:TaskDRLEnvB2ClimbCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TaskDB2PPORunnerCfg",
    },
)

gym.register(
    id="ATEC-TaskD-RL-B2-Drop-v0",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_env_cfg:TaskDRLEnvB2DropCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TaskDB2PPORunnerCfg",
    },
)

gym.register(
    id="ATEC-TaskD-RL-B2-Push-v0",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_env_cfg:TaskDRLEnvB2PushCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TaskDB2PPORunnerCfg",
    },
)

gym.register(
    id="ATEC-TaskD-RL-B2-Mixed-v0",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_env_cfg:TaskDRLEnvB2MixedCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TaskDB2PPORunnerCfg",
    },
)

gym.register(
    id="ATEC-TaskD-RL-B2-Full-v0",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_env_cfg:TaskDRLEnvB2FullCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TaskDB2PPORunnerCfg",
    },
)

gym.register(
    id="ATEC-TaskD-RL-B2-v0",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_env_cfg:TaskDRLEnvB2FullCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TaskDB2PPORunnerCfg",
    },
)

__all__ = ['TaskDEnvCfg', 'TaskDEnvB2Cfg', 'TaskDEnvTron2ALeggedCfg', 'TaskDEnvTron2AWheelCfg', 'TaskDRLEnvB2Cfg', 'TaskDRLEnvB2ClimbCfg', 'TaskDRLEnvB2DropCfg', 'TaskDRLEnvB2PushCfg', 'TaskDRLEnvB2MixedCfg', 'TaskDRLEnvB2FullCfg']
