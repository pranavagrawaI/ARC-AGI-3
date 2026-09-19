# periodic-v1 — screen: Duck alone vs Duck + existing critic on a minimum-spacing schedule, 3600 s

Frozen 2026-09-20 before any push. Prepared through Phase 9 (freeze, scheduler, tests, card);
**no kernel has been pushed and no live evaluation has run.** Launch is a separate decision.

## Hypothesis (frozen)
Removing action-zero reviews and closely spaced no-change refirings may reduce critic
interference while retaining useful later advice. The resulting bundle may outperform Duck alone.

Not claimed by any outcome: that periodic scheduling is optimal; that the trigger tree was *the*
problem; that criticism as such is useful; that K=20 is the right interval. A win supports only
"Duck + this critic, exposed no more often than every ~20 actions, beat Duck alone in this screen."
A loss rejects this bundle, not any single explanation (interval, timing, repeated advice,
evidence, delivery, or trajectory variance all remain live).

## CONTROL = Duck alone (Phase 1 freeze)
| item | value |
|---|---|
| tree | this commit (branch `step2-duck-critic-adapter`); Duck upstream 7652836; prompts byte-identical to upstream without a report (snapshot test) |
| model | `vrfai/Qwen3.6-27B-FP8` on local vLLM, `--reasoning-parser qwen3`, `--max-model-len 65536`, TP 1, RTX PRO 6000 |
| actor sampling | temperature 0.6, top_p 0.95, top_k 20, thinking on, context window 32768, tool steps unbounded (`0`), yield 60 s, python tool timeout 30 s, tool output 1024 tokens |
| harness | `--agent duck-harness --model local --analyzer-timeout 900`, concurrency 10 |
| limits | per-game cap **3600 s** (`--max-runtime-minutes 60`), no action cap; experiment cap 135 min (2 waves × 60 + slack), vLLM startup outside it |
| games | sb26-7fbdac44, su15-1944f8ab, lf52-271a04aa, ft09-0d8bbf25, ls20-9607627b (Step 1/3 manifest) × 4 fresh passes = 20 episodes |
| env | no `DUCK_CRITIC_*` (bundle check: 0 critic keys in `setup_commands.json`) |
| scoring | levels = max level reached − 1 per episode from `events.jsonl`; host-confirmed, within the per-game cap |

Duck is not modified in this commit. The only source changes are inside `inference/critic/`
plus one diagnostic viewer event in `solver.py`'s existing except-branch.

## CANDIDATE = periodic-v1
| item | decision |
|---|---|
| candidate | existing critic (question mode), trigger tree replaced by minimum spacing K=20 |
| configuration | `DUCK_CRITIC_ENABLED=1 DUCK_CRITIC_MODE=question DUCK_CRITIC_TRIGGER=periodic DUCK_CRITIC_PERIOD=20` |
| clock | existing `len(game_run.history)` (`solver.action_count`), automatic RESETs included |
| anchor | moved in `PeriodicTriggerPolicy.decide()` — before packet construction and the model call; a fresh `CriticRunner` per game session starts at 0 |
| invocation | first Duck boundary at which ≥20 actions have passed since the last attempt; no catch-up, no `level_start`, no `stall_*`, no actor request, nothing at action 0 |
| critic behaviour | prompt, packet, "already asked" memory, parsing, fallback-report injection, advisory authority: unchanged (same bytes as Step 3's critic arm) |
| deadline behaviour | existing request bound (seconds left + 20 s grace) only; no late-review heuristic |
| failure | fallback report still injected; an escaped exception is caught by the existing hook, logged as a `critic` viewer event with `error`, and not retried until +20 actions |

Tests: `tests/critic/test_periodic.py` (17) — 0/19 no fire, 20/21/45 fire, 18→45 overshoot fires once
and re-anchors at 45 (64 silent, 65 fires), same count re-entered is silent, level change and
actor request ignored, completion at the due boundary yields one periodic firing, clock is the
action count not the transition stream, fresh anchor 0, env selection, unknown mode rejected,
runner ignores 19 no-change actions, fallback injected, escaped exception logged and not retried.
"Critic succeeds, next actor request fails" is covered by the existing
`test_injected_report_is_logged_in_transcript` (report consumed once even when the call fails).
Full suite 58/58.

## Predictions (written before data)
- **Exposure** (a forecast, to be measured, never used to filter): at 3600 s periodic-v1 produces
  multiple delivered reviews in a substantial fraction of episodes. Step 3 at 1200 s had median
  30.5 actions/episode (5/20 below or at 20); if actions scale with wall time the median is ~90
  and ~4 attempts/episode. Low-exposure episodes stay in the analysis as outcomes of the candidate.
- **Outcome**: control ≈ 20–30 levels (Step 3: 14 at 1200 s), candidate within ±4. I expect a
  wash; the ft09 loss mechanism (re-firing every 3 no-change actions) is removed by construction,
  so a repeat of ft09 −4 would say the content, not the cadence, is what hurt there.
- **Cost**: critic wall < 2 % of cap; turn latency within 25 % between arms.

## Primary outcome and decision rule (precommitted)
Primary: **total distinct levels completed** over 20 episodes per arm, host-confirmed within the
per-game cap. Δ = L_candidate − L_control; Δ_g per game.

**Advance periodic-v1 to 7200 s confirmation iff Δ ≥ 8 and Δ − max_g(Δ_g⁺) > 0.** Otherwise retain
Duck alone (small positive → ambiguous; tie → retain; negative → reject). The threshold is a chosen
minimum improvement worth funding confirmation, not a calibrated statistic; an unexpectedly low
control total does not change it after results arrive.

Secondary: TAAF score, actions, turns, s/turn, critic wall, delivery funnel, uptake, failures.

## Delivery funnel (logs, not scheduling conditions)
`readback.py` derives per attempt, from `events.jsonl` + `transcripts/*_critic.jsonl`:
attempted (`critic` viewer events, incl. escaped) → generated (parsed | fallback) → included
(`[CRITIC REPORT]` with the packet id in that analysis step's transcript) → actor completed
(`[ANALYZER STATUS]` for that step) → subsequent action recorded for that step. Validated on
Step 3's critic arm: 50 → 50 → 50 → 50 → 49 (the one miss is the final turn at the cap).

## Validity / invalidation
- Both arms run contemporaneously on separate kernels (separate GPUs, separate vLLM); no shared
  serving resource, so parallel arms are acceptable. Turn latency differing >25 % → substrate
  differed; rerun.
- Candidate arm with 0 attempts in any episode with ≥ 21 actions → scheduler did not run; invalid.
- Any game ended by a firing → invalid.
- Failed episodes keep their completed levels and are not rerun; a slot that never started is run
  later with the same configuration. No replacement of low-performing trajectories.
- A kernel that errors before its games complete is re-pushed once; nothing is read until both
  arms have 20 episodes.

## Launch (NOT executed — recorded so the push is mechanical)
```
cd ~/projects/arc-agi-3-reference/duck-critic-adapter/ARC3-Inference
source /tmp/duck-step1-env.sh
# control
unset DUCK_CRITIC_ENABLED DUCK_CRITIC_MODE DUCK_CRITIC_TRIGGER DUCK_CRITIC_PERIOD
uv run --no-sync --offline inference-taaf-run \
  --game 'sb26-7fbdac44,su15-1944f8ab,lf52-271a04aa,ft09-0d8bbf25,ls20-9607627b' \
  --environments-dir /kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files \
  --run-name duck-periodic-v1-control --experiment-dir ../runs/periodic-v1/control \
  --max-runtime-minutes 60 --max-experiment-runtime-minutes 135 \
  --n-passes 4 --concurrent-jobs 10 --agent duck-harness --model local --analyzer-timeout 900 \
  --deployment-target kaggle --no-deployment-wait \
  --kaggle-username pranavagrawai --kaggle-kernel-slug taaf-duck-periodic-v1-control --kaggle-kernel-title taaf-duck-periodic-v1-control \
  --kaggle-dataset-ref pranavagrawai/taaf-kaggle-source-duck-periodic-v1-control \
  --kaggle-accelerator NvidiaRtxPro6000 \
  --kaggle-dataset-version-message 'periodic-v1 control: duck alone, 5 games x 4 passes, 3600s cap' \
  --slurm-gpu B200 --slurm-gpu-count 1 --slurm-time 06:00:00
# candidate
export DUCK_CRITIC_ENABLED=1 DUCK_CRITIC_MODE=question DUCK_CRITIC_TRIGGER=periodic DUCK_CRITIC_PERIOD=20
uv run --no-sync --offline inference-taaf-run ... (same, with run-name/slug/dataset-ref
  duck-periodic-v1-candidate and message 'periodic-v1 candidate: duck + critic, periodic K=20, 3600s cap')
```
Pre-launch checks: `--kaggle-dry-run` of both bundles (tree hash, 0 vs 4 critic keys in
`setup_commands.json`, pickled cap 3600 s, 20 episodes / 2 waves); Kaggle token age (< 12 h);
GPU quota ≥ 5 h (two kernels × ~2 h 15).

Readback: `python3 runs/periodic-v1/readback.py control/kaggle-output candidate/kaggle-output`.
