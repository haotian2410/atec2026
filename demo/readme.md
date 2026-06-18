只允许修改demo下面的solution.py和tool以及test目录下的代码。

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
        -> RGB-D / LiDAR 尝试校正投放区位置
        -> 状态机输出底盘速度命令
    -> _extract_policy_obs()
        -> 把底盘速度命令塞进腿部 policy 观测
    -> policy.pt 推理
    -> 输出全身 action

  状态机流程是：

  GO_TO_DROP_STAND
    走到投放区外侧

  FACE_DROP_CENTER
    面向投放区

  TURN_AROUND_SCAN_READY
    原地转 120/180/240 度
    每个方向保存 scan_*_head/ee_rgb.png 和 depth.npy

  READY_SCAN_TRASH
    等待 yolo_results.json
    YOLO runner 检测 scan_* 图片并写 JSON

  APPROACH_TRASH_TARGET
    solution.py 持续保存 live_* 当前帧
    YOLO runner --watch 持续检测 live_* 图片并更新 JSON
    本体相机 live head/video 最近目标优先
    没有本体目标时用 ee 粗靠近

  READY_TO_GRASP
    底盘静止，等待你接机械臂抓取逻辑


  YOLO runner 的作用时机

  taskb_yolo_runner.py --watch 会循环做：

  找最新 outputs/taskb_scan/<时间戳>/
  读取 scan_*_rgb.png 和最新 live_*_rgb.png
  用 best.pt 检测
  读取匹配 depth.npy
  写 outputs/taskb_scan/<时间戳>/yolo_results.json

  solution.py 会每 10 步读取这个 JSON。

  注意点

  - --model /path/to/best.pt 要换成你的真实模型路径。
  - 如果你在主任务创建 scan 目录前启动 YOLO runner，它可能找不到目录；最简单是主任务开始跑后再开
    YOLO。

  - pose_viewer.py 和 lidar_height_scan_check.py --plot 都是调试显示，不参与控制。
  - LiDAR --plot 是静态显示当前最近一帧；想刷新效果，需要关掉再开，或者后面再加 live 刷新模式。

