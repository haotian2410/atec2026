from isaaclab.utils import configclass  # IsaacLab 配置类装饰器，让该类可以作为 Hydra/RSL-RL 配置入口。
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg  # RSL-RL runner、网络和 PPO 算法配置。


@configclass  # 声明这是可被 IsaacLab 注册系统解析的配置类。
class TaskDB2PPORunnerCfg(RslRlOnPolicyRunnerCfg):  # TaskD plain B2 专用 PPO runner 配置。
    num_steps_per_env = 32  # 每个并行环境一次 rollout 收集 32 步；TaskD 有接触/推箱子，略长 rollout 有利于估计回报。
    max_iterations = 12000  # 默认训练 12000 个 PPO iteration；可用 --max_iterations 覆盖。
    save_interval = 250  # 每 250 个 iteration 保存一次 checkpoint，方便回滚和挑选模型。
    experiment_name = "task_d_b2_box_step"  # 日志目录名：logs/rsl_rl/task_d_b2_box_step/。
    empirical_normalization = False  # 不启用 RSL-RL 经验归一化；当前观测已做基础 scale/clip，先保持简单可控。
    policy = RslRlPpoActorCriticCfg(  # actor-critic 网络结构配置。
        init_noise_std=0.3,  # 初始动作高斯噪声标准差；plain B2 早期训练先降低动作随机性。
        actor_obs_normalization=False,  # actor 不额外做 obs normalization，避免与 lidar clip/scale 重叠。
        critic_obs_normalization=False,  # critic 不额外做 obs normalization，保持配置行为透明。
        actor_hidden_dims=[512, 256, 128],  # actor MLP 三层隐藏层；输入包含大量 lidar scan，需要较大第一层。
        critic_hidden_dims=[512, 256, 128],  # critic MLP 与 actor 同规模，保证足够表达接触/阶段价值。
        activation="elu",  # 使用 ELU 激活，沿用项目里 B2 locomotion 配置的稳定选择。
    )
    algorithm = RslRlPpoAlgorithmCfg(  # PPO 算法超参数配置。
        value_loss_coef=1.0,  # value loss 权重，保持 RSL-RL 常用默认值。
        use_clipped_value_loss=True,  # 对 value loss 使用 clipping，减少 critic 更新过猛。
        clip_param=0.2,  # PPO policy ratio clip 范围，标准 PPO 设置。
        entropy_coef=0.006,  # 熵奖励权重；保留探索，但低于普通行走，避免动作过乱。
        num_learning_epochs=5,  # 每批 rollout 重复训练 5 个 epoch。
        num_mini_batches=4,  # 将 rollout 切成 4 个 mini-batch 更新。
        learning_rate=5.0e-4,  # 学习率；比 B2 flat 的 1e-3 更保守，适合接触丰富任务。
        schedule="adaptive",  # 根据 KL 自适应调整学习率，是 RSL-RL 常用稳定设置。
        gamma=0.99,  # 折扣因子，任务 D 需要考虑数秒后的过障奖励，使用 0.99。
        lam=0.95,  # GAE lambda，平衡偏差和方差。
        desired_kl=0.01,  # 目标 KL；adaptive schedule 会围绕它调学习率。
        max_grad_norm=1.0,  # 梯度裁剪上限，防止接触奖励导致梯度爆炸。
    )
