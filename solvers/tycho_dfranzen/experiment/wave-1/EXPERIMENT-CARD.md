# Wave-1 experiment card, frozen 2026-10-05 before launch

Supersedes pair-1. Pranav stopped pair-1 while both arms were QUEUED (zero GPU spent) after the
pre-run audit found a crash path and missing affordances. He then asked for lean vs port vs Duck,
using the available GPU hours with more and faster experiments.

## Question
Does Tycho on dfranzen's runtime beat dfranzen's own Duck solver once it has dfranzen's
complementary affordances? And do executable world models (Builder) plus Scribe earn their cost,
or would the Tycho Actor alone do better?

## Arms (same weights, drafter, SGLang server, sampling, GPU, games and caps)
| Arm | Solver | Mode / treatment |
|---|---|---|
| control | dfranzen M2 notebook, unchanged solver | Duck harness |
| port | Tycho hybrid + port (below) | orchestrator / control: Actor + Builder + Scribe |
| lean | the port notebook, one env cell changed | no_world_model / no_scribe: Actor only |

Port relative to pair-1's candidate:
1. GAME_OVER legality fix. The engine raises on any non-RESET after GAME_OVER, yet still lists the game actions (verified on ls20). The pair-1 hybrid host would mark that game crashed; now only RESET is offered.
2. Task statement in the system prompt. The NOOA Actor never received canonical Tycho's objective, action economy or RESET semantics. Added together with dfranzen's flag semantics, bar rule and level-transfer system text.
3. Event guidance in observations: dfranzen's game-over diagnosis (adapted to Tycho's workspace and wmlib.death_events), level-transfer prompt, border-only change note, and the reason a queued batch stopped.
4. Batch no-op guard tests the interior (4-cell border excluded, dfranzen's gameplay_changed). q6 ls20 had 11 border-only batch continuations the whole-board test let through.

Deliberately not ported, with grounds: stale-state guard (Tycho takes one action per decision, so no snippet keeps acting); terminal guard (Tycho batches already stop at boundaries and terminals); function retention (workspace modules); repeated-opener labels and token yield (no Duck re-prompt loop; dfranzen found yield neutral); UNDO and animations (already present, and animations used in q6); frame_diff (q6 never called the dfr helpers, and Tycho already shows a cell diff plus a diff image).

## Fixed budget (src/hybrid/pair.py, embedded verbatim in every arm)
Games: cn04, dc22, ka59, ls20, m0r0, r11l, re86, sc25, su15, vc33. Chosen before any wave-1 data as the public games where both historical controls neither finished nor stayed at zero by 3600 s. All 10 count.
10 concurrent games (10 streams, the server's saturation and dfranzen's submission setting), one pass, 3600 s per game, no action cap.
Session hard cap 4800 s, soft stop 4500 s, watchdog force at 4790 s. Reservation 3 x 4800 s = 4.0 GPU-hours.
Account quota on 2026-10-05: 21,600 s per week, about 4,845 s used, resets 2026-10-10. Remaining about 4.65 h, so this wave uses it nearly fully.
Launch: frozen draw (seed 3726991263) = port, control, then lean once a slot frees (Kaggle allows two GPU sessions).

## Measurements
Primary: per-game official score and levels at session end. Secondary: total levels, actions,
completion tokens per arm and per role, levels per million completion tokens, crashes and stop reasons.
Mechanism report (port): Builder offered, requested, admitted, completed, verified model, prospective prediction, correctness. Plus game-over count and RESET handling.

## Noise and result map (precommitted)
Identical controls (control vs control-r2, 25 games, 3600 s) differ per game with sd 1.6 levels.
The sum over 10 games of independent single-run differences then has sd of about 5 levels.

For each contrast (port vs control, lean vs control, lean vs port), with Δ = total levels:
- Δ >= +10 (about 2 sd): BETTER, eligible for a replicate when quota returns.
- Δ <= -10: WORSE. Diagnose from traces; do not iterate that arm blindly.
- Otherwise INCONCLUSIVE at n=1. Report Δ, the per-game sign count, and levels per M tokens.

Builder question: lean vs port is the executable-world-model-plus-Scribe effect at equal compute.
- If lean >= port + 10: the Builder and Scribe cost more than they buy at this load.
- If port >= lean + 10: they earn their cost.
- Otherwise the effect is smaller than about 10 levels per 10 games here, and replication decides.

## Invalidity
Changed hardware, weights, caps or games. A crashed game (the game-over fix is under test). Missing receipts or unprovable actions. A third concurrent session. An overlapping run that contaminates the quota delta. Outcome and spend are kept either way, and nothing is retried silently.

## Frozen identity
pair-identity.json: payload ad2acba3cbd962968a462a196061c9b961cb53de253174aca9b920df23bb6041 (source-v8, read back byte-identical), pair.py 25487b9a...
Local qualification: 49 checks on a scratch copy and verify.py (LOCAL-QUALIFICATION.json).
Unqualified until GPU: the hardened per-game-UID host at 10 concurrent games, real-model use of the new guidance, and hybrid throughput at 10 streams.
