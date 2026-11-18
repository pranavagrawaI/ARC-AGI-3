# Q6 executable prediction feedback: bounded longer run

Self-contained candidate for `pranavagrawai/q6-prediction-feedback-20260906`. Adds model-consistency feedback to the hardware-qualified q6 host. Inference configuration remains baseline FP8, thinking disabled,4096 output tokens. Read EXPERIMENT-CARD.md for interpretation and stopping rules.

One ls20 game,100 individual Solver actions maximum. Gameplay admission starts on the first playable action and stops after7200s; a pending decision can overrun by up to900s. Normal official cleanup preserves model accounting. Launcher kill envelope8400s and Kaggle session10800s bound total use to3GPU-hours. A hard timeout is failure. `solver-workspaces/game-stops.jsonl` distinguishes `wall_budget`, `step_budget`, and `win`; session `stop.json` remains the separate lifecycle reason `competition_agent_cleanup`.

After source edits,freeze/rebuild and run local qualification before launching:

```bash
python3 deployment/kaggle/q6-prediction-feedback-20260906/freeze_candidate.py
python3 deployment/kaggle/q6-prediction-feedback-20260906/prepare.py
python3 deployment/kaggle/q6-prediction-feedback-20260906/prepare.py --check
/home/pranav/.local/share/uv/tools/kaggle/bin/python deployment/kaggle/q6-prediction-feedback-20260906/verify_probe.py
kaggle quota --format json
```

Only after passing quick qualification,push once using LAUNCH.json argv. Monitor with `kaggle kernels status pranavagrawai/q6-prediction-feedback-20260906`. Download terminal evidence using:

```bash
kaggle kernels output pranavagrawai/q6-prediction-feedback-20260906 --path analysis/deployment-probes/20260906-q6-prediction-feedback/raw
python3 deployment/kaggle/q6-prediction-feedback-20260906/assess.py analysis/deployment-probes/20260906-q6-prediction-feedback/raw/nooa-join-public3 --output analysis/deployment-probes/20260906-q6-prediction-feedback/SCORE.json
```

Assessor PASS concerns host evidence,not a solver-improvement verdict. No inference optimization,no automatic retry,and no competition submission. Do not claim the summarizer was exercised unless actual calls establish it.
