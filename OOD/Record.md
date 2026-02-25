# Record: SafeDreamer_v2 paper_sprint (seed completion + diagnostics)

## Scope and constraints
- Repo: /root/autodl-tmp/projects/SafeDreamer_v2
- Experiments: /root/autodl-tmp/experiments/safedreamer_v2/paper_sprint
- Env: conda env `safedreamer` (use `conda run -n safedreamer`), CUDA 11.8 already present.
- Required env vars: CUDA_VISIBLE_DEVICES=0, MUJOCO_GL=egl, JAX_PLATFORM_NAME=gpu.
- Do not reinstall torch/cuda.
- Required pipeline: train seeds 0-4 (4 configs), eval, build tables/figures/defense docs, update ppt.md, decision report, then shutdown.

## Key files and modifications
- /root/autodl-tmp/projects/SafeDreamer_v2/scripts/run_paper_longtrain.sh
  - Added SEEDS support (override default seeds).
  - Added skip logic: if checkpoint exists for config/seed, skip.
  - Merge previous status entries outside selected SEEDS into new status file.
  - Ensured MUJOCO_GL=egl is exported inside script.
- GPU precheck saved at:
  - /root/autodl-tmp/experiments/safedreamer_v2/paper_sprint/manifests/paper_gpu_precheck_latest.txt

## Observations and diagnostics (chronological)
- Initial state: paper_train_status_latest.tsv had only seeds 0/1 completed.
- Training attempt for baseline seed3 repeatedly appeared to “stall” at step=3005 in metrics.jsonl.
  - Evidence: multiple run dirs for v2_baseline seed3 with max_step=3005.
  - Example run dir: /root/autodl-tmp/experiments/safedreamer_v2/paper_sprint/runs/20260223-171229_v2_baseline_safetygymcoor_SafetyPointGoal1-v0_3
  - Example metric: metrics.jsonl max_step=3005, timer/duration~58s.
  - Logs in manifests only had the CMD line; err logs empty.
- Root cause of “stuck metrics” was misinterpretation:
  - In SafeDreamer/embodied/core/when.py, `Clock` uses wall time; `log_every` is seconds, not steps.
  - Therefore metrics.jsonl can appear unchanged for long periods even when training continues.
  - Training progress is reflected by replay chunk files under `run_dir/replay`.

## Process management and monitoring
- Training was moved into tmux session: `paper_train_seed34`.
- Monitor loop (5 min interval) writes to:
  - /root/autodl-tmp/experiments/safedreamer_v2/paper_sprint/manifests/paper_train_monitor_seed34.log
- Latest monitor entries (2026-02-25 ~03:16 CST) show:
  - GPU utilization 0%, memory 0 MiB.
  - No running SafeDreamer/train.py process.
  - Latest replay chunk timestamps are 2026-02-24 07:07–07:08.

## Current training status (as of latest check)
- file: /root/autodl-tmp/experiments/safedreamer_v2/paper_sprint/manifests/paper_train_status_latest.tsv
- Total rows: 17 successes.
- Completed (success):
  - v2_baseline: seeds 0,1,2,3,4
  - v2_u_only: seeds 0,1,3,4
  - v2_cert_only: seeds 0,1,3,4
  - v2_full: seeds 0,1,3,4
- Missing (not present in status, no run dir found):
  - v2_u_only seed2
  - v2_cert_only seed2
  - v2_full seed2
- No training process currently running (`pgrep -af SafeDreamer/train.py` returns none).

## Understanding of why missing seeds exist
- Seed2 was completed for v2_baseline only.
- When the script was re-run to complete the remaining seeds, it was run with SEEDS="3 4" to avoid the earlier baseline seed3 bottleneck.
- This allowed all configs to complete seeds 3/4 but left seed2 for v2_u_only / v2_cert_only / v2_full unexecuted.

## What this means (current state)
- Training is not running and the required 4 configs x 5 seeds matrix is incomplete.
- Downstream steps (eval, tables/figures/defense docs, ppt updates, decision report) should not proceed until missing seed2 runs exist.

## Proposed fix (not executed)
- Run only missing seed2 for the three configs (do NOT rerun existing seeds):
  - Use `SEEDS="2"` and rely on skip logic for existing checkpoints.
  - Command (do not execute here):
    - `AUTO_SHUTDOWN=0 SEEDS="2" STEPS=120000 EVAL_EVERY=5000 LOG_EVERY=1000 SAVE_EVERY=10000 bash scripts/run_paper_longtrain.sh`
- After completion, verify that paper_train_status_latest.tsv has 20 success rows.

## Evidence locations
- Training status: /root/autodl-tmp/experiments/safedreamer_v2/paper_sprint/manifests/paper_train_status_latest.tsv
- Monitor log: /root/autodl-tmp/experiments/safedreamer_v2/paper_sprint/manifests/paper_train_monitor_seed34.log
- Example seed3 baseline run dirs:
  - /root/autodl-tmp/experiments/safedreamer_v2/paper_sprint/runs/20260223-174210_v2_baseline_safetygymcoor_SafetyPointGoal1-v0_3
- Latest seed4 full run dir:
  - /root/autodl-tmp/experiments/safedreamer_v2/paper_sprint/runs/20260224-052455_v2_full_safetygymcoor_SafetyPointGoal1-v0_4

