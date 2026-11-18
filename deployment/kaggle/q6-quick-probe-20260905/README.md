# Q6 repaired-host quick probe

Prepared for `pranavagrawai/q6-quick-host-20260905`. No push performed.

One ls20 game, control arm, 10 individual environment actions, 4096 output tokens, thinking disabled, Kaggle FP8/local inference. Request 180s; worker 900s; gameplay 1800s; session 3600s; at most one GPU-hour. One separately authorized private push, no automatic retries or competition submission. Live quota must be checked before launching.

The notebook is self-contained. Its source snapshot lives here and does not import the mutable q6 candidate at runtime. DONOR-PIN.json records the accounting-fix snapshot used; CHASSIS.json binds the repaired actor/host and outer runtime sources; LAUNCH.json binds notebook, metadata, card and preparation/assessment tooling.

Changes from the speed probe:

- NOOA session schema v4 carries per-episode request evidence and composes with the owned runtime reconciler. Actor and summarizer calls both pass recording, reasoning and integrity guards.
- Frequency-derived HUD rows are diagnostic only. They cannot declare an action inert, remove rows from inert-state identity or skip queued actions. Literal no-change, level and terminal stops remain effective. The historical step5 move no longer becomes a false wall hit.
- Per-step trajectory reads exact before/after host transitions. Unavailable animation frames are explicit.
- Dispatch uses notifications as wakeups and delivers the latest committed state. Duplicate stale-only notifications are suppressed; new user messages survive. Incoming/latest/delivered identities are logged.
- Host batch stops, skipped actions, unexecuted tails, actor batches, individual actions, calls by source and detector eligibility are reported separately.

No mandatory experiment workflow, new prediction predicate, reasoning-on mode or server-abort change is included. Changed pixels still use the existing coarse `move` outcome name; this does not prove object movement. A ten-action run has zero stall eligibility under the unchanged q6 threshold. It is excluded from the research arm and is not a single-factor speed comparison.

Local qualification: **20 passed**, including extracted-payload replay, real NOOA actor plus summarizer with a fake model, reasoning rejection, actual worker/headless state delivery, GAME_OVER and level boundaries, exact budget/cleanup, and missing/extra request rejection. The installed Kaggle 2.2.4 request constructor passed with zero transport calls. Local tests needed execution outside the restricted sandbox because Python default-executor shutdown hung there after assertions completed; the same test passed outside it. No paid model request or GPU run was used for qualification.

From the repository root:

```bash
python3 deployment/kaggle/q6-quick-probe-20260905/prepare.py --check
/home/pranav/.local/share/uv/tools/kaggle/bin/python deployment/kaggle/q6-quick-probe-20260905/verify_probe.py
env MPLCONFIGDIR=/tmp/q6-quick-mpl LITELLM_LOCAL_MODEL_COST_MAP=True deployment/kaggle/.venv/bin/python -m pytest analysis/deployment-probes/20260905-q6-quick-host/test_payload.py analysis/deployment-probes/20260905-q6-quick-host/test_join_payload.py -q
```

After a separately authorized run and download:

```bash
python3 deployment/kaggle/q6-quick-probe-20260905/assess.py analysis/deployment-probes/20260905-q6-quick-host/raw --output analysis/deployment-probes/20260905-q6-quick-host/SCORE.json
```

The quick assessor checks host qualification without the research scorer's >=15-turn or >=3-stall requirements. It correctly rejects the old speed artifact for its known schema, state-delivery and false-no-impact failures. GPU startup, real summarizer incidence and actual FP8 behavior remain live checks. Full local evidence is in analysis/deployment-probes/20260905-q6-quick-host/.
