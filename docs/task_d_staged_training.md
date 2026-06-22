# Task D staged RSL-RL training commands

These commands are for the fixed-map Task D strategy:

1. push the box to the platform side
2. climb onto the box
3. climb onto the platform
4. drop down
5. walk across the finish line

Do not start these long Isaac Sim training runs automatically from automation or review scripts. Run them manually on an Ubuntu workstation with Isaac Sim / Isaac Lab configured.

## Setup

Run from the repository root:

```bash
cd /home/lht/code/ATEC2026
```

`scripts/rsl_rl/train_task_d.py` keeps its default behavior: if `--task` is omitted it trains `ATEC-TaskD-RL-B2Piper-Climb-v0`, and if the baseline exists it uses `atec_robot_model/baseline/unitree_b2_flat/policy.pt` as a weak pretrained initialization.

All stages write checkpoints under:

```text
logs/rsl_rl/task_d_b2_piper_box_step/<RUN_DIR>/model_<ITER>.pt
```

For resume commands, replace `<RUN_DIR>` with the run directory name under `logs/rsl_rl/task_d_b2_piper_box_step/`, and replace `<CHECKPOINT>` with a checkpoint file name such as `model_2500.pt`.

## Stage 1: push

Start from the walking baseline and train the box push stage:

```bash
python scripts/rsl_rl/train_task_d.py \
  --task ATEC-TaskD-RL-B2Piper-Push-v0 \
  --run_name task_d_push \
  --num_envs 1024 \
  --max_iterations 3000
```

Resume push training explicitly from a saved checkpoint:

```bash
python scripts/rsl_rl/train_task_d.py \
  --task ATEC-TaskD-RL-B2Piper-Push-v0 \
  --run_name task_d_push_resume \
  --num_envs 1024 \
  --max_iterations 3000 \
  --resume \
  --load_run <PUSH_RUN_DIR> \
  --checkpoint <PUSH_CHECKPOINT>
```

## Stage 2: climb

Start climb from the best push checkpoint:

```bash
python scripts/rsl_rl/train_task_d.py \
  --task ATEC-TaskD-RL-B2Piper-Climb-v0 \
  --run_name task_d_climb_from_push \
  --num_envs 1024 \
  --max_iterations 4000 \
  --resume \
  --load_run <PUSH_RUN_DIR> \
  --checkpoint <PUSH_CHECKPOINT>
```

If you want the original default climb training instead, this remains valid:

```bash
python scripts/rsl_rl/train_task_d.py
```

## Stage 3: drop

Start drop from the best climb checkpoint:

```bash
python scripts/rsl_rl/train_task_d.py \
  --task ATEC-TaskD-RL-B2Piper-Drop-v0 \
  --run_name task_d_drop_from_climb \
  --num_envs 1024 \
  --max_iterations 3000 \
  --resume \
  --load_run <CLIMB_RUN_DIR> \
  --checkpoint <CLIMB_CHECKPOINT>
```

## Stage 4: mixed

Use the mixed Task D gym id for reset-level consolidation, initialized from the best drop checkpoint:

```bash
python scripts/rsl_rl/train_task_d.py \
  --task ATEC-TaskD-RL-B2Piper-Mixed-v0 \
  --run_name task_d_mixed_from_drop \
  --num_envs 1024 \
  --max_iterations 3000 \
  --resume \
  --load_run <DROP_RUN_DIR> \
  --checkpoint <DROP_CHECKPOINT>
```

## Stage 5: full

Continue full end-to-end training from the best mixed checkpoint:

```bash
python scripts/rsl_rl/train_task_d.py \
  --task ATEC-TaskD-RL-B2Piper-Full-v0 \
  --run_name task_d_full_from_mixed \
  --num_envs 1024 \
  --max_iterations 6000 \
  --resume \
  --load_run <MIXED_RUN_DIR> \
  --checkpoint <MIXED_CHECKPOINT>
```

The alias `ATEC-TaskD-RL-B2Piper-v0` also maps to the full Task D RL environment, but prefer `ATEC-TaskD-RL-B2Piper-Full-v0` in staged logs so the stage is clear.

## Play and export

`scripts/rsl_rl/play.py` loads a checkpoint, runs policy playback, and exports both:

```text
logs/rsl_rl/task_d_b2_piper_box_step/<RUN_DIR>/exported/policy.pt
logs/rsl_rl/task_d_b2_piper_box_step/<RUN_DIR>/exported/policy.onnx
```

Play and export a chosen full checkpoint:

```bash
python scripts/rsl_rl/play.py \
  --task ATEC-TaskD-RL-B2Piper-Full-v0 \
  --num_envs 16 \
  --load_run <FULL_RUN_DIR> \
  --checkpoint <FULL_CHECKPOINT> \
  --real-time
```

Record a short playback video while exporting:

```bash
python scripts/rsl_rl/play.py \
  --task ATEC-TaskD-RL-B2Piper-Full-v0 \
  --num_envs 16 \
  --load_run <FULL_RUN_DIR> \
  --checkpoint <FULL_CHECKPOINT> \
  --video \
  --video_length 600
```
