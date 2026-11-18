This isolates executable model repair using actual q6 CodeAct and saved observations. It executes no environment actions. Four starts use the same history through step 6; steps 7–10 stay outside the Actor workspace and are used only by the independent scorer. Both receive the same explicit repair instruction. The candidate additionally receives automatically calculated model feedback and its new action gate. This is a favorable diagnostic of correction under instruction, not an autonomous discovery or level-solving benchmark.

Run offline checks with the NOOA environment:

```bash
/home/pranav/projects/arc-agi-3-reference/nooa/.venv/bin/python -m pytest -q analysis/deployment-probes/20260906-q6-prediction-feedback/test_probe.py
```

Create a new output identity and execute the bounded paid replay:

```bash
/home/pranav/projects/arc-agi-3-reference/nooa/.venv/bin/python analysis/deployment-probes/20260906-q6-prediction-feedback/probe.py --run /tmp/q6-feedback-unique-run --execute
```

Omit `--execute` to prepare a disposable dry-run fixture without sending requests; that identity cannot later be reused. All source paths are hashed before execution and checked before each arm. The relay holds credentials outside the Actor worker. Raw requests/responses retain tool calls, provider metadata, token usage and reported cost; inspect these alongside SCORE.json. The $1 cap is a conservative reservation using byte-count input bounds and a provider price ceiling, not a post-hoc bill cutoff. No paid retry occurs after failure; source or cap changes require a fresh identity/card.
