# periodic-v1 launch — 2026-09-19 20:45–20:46Z

Both arms pushed from tree `baf01a8` (card frozen at `03e69bd`, unchanged since):

| arm | kernel | pushed | env in setup |
|---|---|---|---|
| control | `pranavagrawai/taaf-duck-periodic-v1-control` v1 | 20:45Z | no `DUCK_CRITIC_*` |
| candidate | `pranavagrawai/taaf-duck-periodic-v1-candidate` v1 | 20:46Z | `ENABLED=1 MODE=question TRIGGER=periodic PERIOD=20` |

5 games × 4 passes, 3600 s/game, concurrency 10, 2 waves, experiment cap 135 min, kernel cap
8100 s, RTX PRO 6000 each. Expected: ~10 min vLLM startup + ~2 × 60 min → COMPLETE ~23:05Z.
Token refreshed 20:43Z (expires 08:43Z). The launcher env file was rebuilt from the recorded
recipe after `/tmp` was cleared; its values were checked against the Step 3 bundle
(ctx 32768, yield 60, upscale 4, top_k 20) before push, and a copy now lives at
`~/.cache/duck-step1-env.sh`.

Readback when both are COMPLETE:
`python3 runs/periodic-v1/readback.py runs/periodic-v1/control/kaggle-output runs/periodic-v1/candidate/kaggle-output`
