# flash-next — submission candidate for 2026-09-25 (frozen before push)

## Why
Board best is ours at 1.58 (arm F, dense Qwen3.8-27B-FP8). Keith Tyser's public Duck +
Qwen3.8-Flash-Next NVFP4 / MTP3 notebook reports 3.38 on the same harness family (TAAF duck-harness,
ARC3-Inference). Source: tall-humerus analysis/kaggle-approaches-20260925/REPORT.md.
His `serving_setup.py` pins `LOCAL_ANALYZER_YIELD_SECONDS=60` — the yield that ended 45 % of our 3.8
turns with no action — and has neither prompt fix.

## Arms (25 public games, 1 pass, 28-way, 7920 s/game, Keith's notebook gates)
| arm | kernel | source bundle | change |
|---|---|---|---|
| K  | pranavagrawai/duck-flash-next-k  | keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1 (as-is) | none: reproduction/control |
| KF | pranavagrawai/duck-flash-next-kf | pranavagrawai/duck-flash-next-kf-src | K + arm F: tool_agent.py and inference/critic from runs/qwen38-fixes/f bundle (critic off), notebook sets YIELD 300, GAMEOVER_RETRY=1, LEVELUP_REVIEW=1 after setup, before the benchmark pickle imports tool_agent |

serving_setup.py is untouched in KF (it self-hash-checks against SOURCE_IDENTITY.json).
Local check: KF tree imports in isolation, flags resolve to 300/True/True, benchmark unpickles (duck-harness, 7920 s, 28).

## Mechanism checks (KF)
- Log line `FIXES_ENV {...}` printed.
- Yielded-turn share in KF < K's (the yield fix on this stack).

## Submission rule (one slot/day; needs Pranav's go)
Levels on 25 public games, sd of a delta ~4:
- Both ran: submit **KF** unless K beats KF by >= 5 levels; then submit K.
- Only one ran to completion: that one.
- Neither: no candidate today; diagnose from logs.
Also record Keith's score.json from each arm (his local public-demo figure was 6.76; different measurement).

## Cost
Two RTX PRO 6000 sessions, ~2.5–3 h each (135 GB model load + one 7920 s wave). 0 submission slots.

## Status 2026-09-25 15:19Z
Dataset uploaded and verified (marker, watchdog, patches, F tool_agent 94300 B). Both kernel pushes
refused: "Maximum weekly GPU quota of 30.00 hours reached." Quota resets Saturday 00:00 UTC.
Launch with `./launch.sh` after reset. Nothing ran; no GPU hours spent on this card.
