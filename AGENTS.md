# ATEC2026 Task D project instructions

## Project goal

We are working on ATEC2026 Task D for Unitree B2 Piper in Isaac Sim / Isaac Lab.
The current strategy is to overfit the fixed competition map:

robot start -> push box to platform side -> climb onto box -> climb onto platform -> drop down -> walk across finish line.

## Important files

- source/atec_rl_lab/atec_rl_lab/tasks/task_d/env_cfg.py
- source/atec_rl_lab/atec_rl_lab/tasks/task_d/rl_env_cfg.py
- source/atec_rl_lab/atec_rl_lab/tasks/task_d/mdp/rewards.py
- source/atec_rl_lab/atec_rl_lab/tasks/task_d/mdp/terminations.py
- scripts/rsl_rl/train_task_d.py
- scripts/rsl_rl/train.py
- scripts/rsl_rl/play.py
- demo/solution.py

## Development rules

- Focus only on Task D unless explicitly asked.
- Use Chinese for subsequent user-facing replies unless explicitly asked otherwise.
- Do not change Task A/B/C behavior.
- Prefer minimal, reviewable changes.
- Do not start long Isaac Sim training runs automatically.
- For heavy training commands, write scripts or commands and ask the user to run them.
- Keep the fixed-map strategy as the first priority.
- Prefer curriculum reset stages over adding more dense rewards.
- When modifying reward logic, check for reward hacking or per-step reward farming.
- When modifying observations/actions, verify compatibility with RSL-RL and demo/solution.py export.
- After changes, run lightweight syntax/import checks when possible.
