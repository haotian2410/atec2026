只允许修改demo下面的solution.py和tool以及test目录下的代码。
python scripts/rsl_rl/train.py \
  --task ATEC-Isaac-LowStance-Flat-Unitree-B2Piper-v0 \
  --headless \
  --video \
  --num_envs 128 \
  --max_iterations 1000

python scripts/rsl_rl/train.py \
  --task ATEC-Isaac-LowStance-Flat-Unitree-B2Piper-v0 \
  --headless \
  --video \
  --num_envs 512 \
  --max_iterations 3000

python scripts/rsl_rl/train.py \
  --task ATEC-Isaac-Velocity-Flat-Unitree-B2WPiper-v0 \
  --headless \
  --video

python scripts/rsl_rl/train.py \
  --task ATEC-Isaac-Velocity-Flat-Unitree-B2WPiper-v0 \
  --headless \
  --num_envs 1024 \
  --max_iterations 12000 \
  --resume \
  --load_run 2026-06-20_19-47-23 \
  --checkpoint model_5999.pt

  推荐启动顺序

  1. 先启动主任务：

PYTHONPATH=$PWD python scripts/play_atec_task.py \
--task ATEC-TaskB-B2Piper \
--enable_cameras \
--debug

  它会运行 demo/solution.py，并生成：

  outputs/taskb_scan/<时间戳>/
  outputs/taskb_pose_state.json
  outputs/taskb_lidar_debug/latest_lidar_debug.npz

  2. 再启动 YOLO runner：

conda activate atec-yolo
cd ~/lht/ATEC2026

python demo/tool/taskb_yolo_runner.py \
--model models/hf/esapzoi_litter_yolov8/best.pt \
--watch


  不写 --scan-dir 时，它会自动找最新的：

  outputs/taskb_scan/*

  所以要等主任务至少创建出扫描目录后再启动。否则会提示找不到 scan directory。

  3. 再启动二维定位可视化：

python demo/test/pose_viewer.py --watch-save

  它读取：

  outputs/taskb_pose_state.json

  可以在主任务启动后任意时间打开。

  4. LiDAR 图建议等主任务跑出一次 LiDAR debug 后再打开：

python demo/test/lidar_height_scan_check.py --plot

  它读取：

  outputs/taskb_lidar_debug/latest_lidar_debug.npz

  如果太早打开，会提示没有 npz。

  运行时调用流程

  主任务启动后，solution.py 每帧执行 predicts()：

  predicts()
    -> 读取 obs["proprio"], obs["image"], obs["extero"]
    -> _compute_navigation_command()
        -> Localizer2D 用 proprio 积分自身位置
        -> RGB-D / LiDAR 尝试做小权重投放区校正
        -> 保存 live RGB-D 帧并轮询 YOLO JSON
        -> 状态机输出底盘速度命令
    -> _extract_policy_obs()
        -> 把底盘速度命令塞进腿部 policy 观测
    -> policy.pt 推理
    -> _map_policy_action_to_env_action() 合成腿部和机械臂 action

  代码边界：

  - solution.py 负责状态机、目标锁定、YOLO 新鲜度保护和动作合成。
  - tool/localizer2d.py 负责策略层 2D 位姿。
  - tool/drop_zone.py 负责橙色投放区 RGB-D 圆环估计；缺 head 图时不会把 video 当 head 外参使用。
  - tool/lidar_height_scan.py 负责 LiDAR height_scan 单帧伪点云诊断/校正。
  - tool/piper_kinematics.py 负责 Piper FK。
  - tool/yolo_targets.py 负责 YOLO JSON 契约、深度补齐、TrashTarget 和伺服速度。
  - tool/taskb_yolo_runner.py 负责扫描目录检测和持续写 yolo_results.json。

  状态机流程是：

  GO_TO_DROP_STAND / FACE_DROP_CENTER
    走到投放区外侧并面向投放区

  FACE_NEAR_SWEEP_START / NEAR_SWEEP_TRASH
    转到背向投放区的有效扇区起点，低速近扫
    只允许本体近目标或很近的 ee 目标接管

  FAR_SEARCH_TRASH
    近处没目标后，用 ee 顺时针扫远处垃圾
    远目标锁定后进入粗接近，避免多个垃圾间来回跳

  APPROACH_TRASH_TARGET
    solution.py 持续保存最新 live_* 当前帧
    YOLO runner --watch 持续检测 live_* 图片并更新 JSON
    每个相机默认只保留最新 live 图参与闭环
    本体相机 live head/video/body 近目标优先，没有本体目标时用 ee 粗靠近
    hold_for_grasp 只是慢速微调限幅，不会让状态机原地硬等
    ready_to_grasp 必须有效深度、小横向误差，并且同一目标连续 3 张不同 live 图确认

  PRONE_TRANSITION / READY_TO_GRASP
    ready 确认后先平滑趴下，趴下后保持底盘静止
    READY_TO_GRASP 仍是机械臂抓取逻辑的接入点


  YOLO runner 的作用时机

  taskb_yolo_runner.py --watch 会循环做：

  找最新 outputs/taskb_scan/<时间戳_纳秒>/
  读取 scan_*_rgb.png 和最新 live_*_rgb.png
  用 best.pt 检测
  读取匹配 depth.npy
  写 outputs/taskb_scan/<时间戳_纳秒>/yolo_results.json，meta.updated_at 表示写入时间

  solution.py 每 3 步读取这个 JSON。若 age_s 超过阈值或 live step 落后过多，会清空目标并停车；连续 stale 超过阈值后按退避重启 YOLO runner。

  注意点

  - --model /path/to/best.pt 要换成你的真实模型路径。
  - TASKB_YOLO_MAX_AGE_S、TASKB_YOLO_MAX_LIVE_LAG_STEPS、TASKB_YOLO_STALE_RESTART_COUNT、TASKB_YOLO_RESTART_BACKOFF_S 可调检测失效保护。
  - pose_viewer.py 和 lidar_height_scan_check.py --plot 都是调试显示，不参与控制。
  - 纯逻辑单测在 demo/test/test_taskb_logic.py，可用 python -m unittest demo.test.test_taskb_logic 运行。

