# wave-2
Frozen 2026-10-06 before launch. Supersedes wave-1 (never launched). The source is
tycho-dfranzen-source-v9 (README "v5 lean Actor"); the identity is in pair-identity.json.

Question: Once the Builder and Scribe are removed and closed decisions are pruned, can Tycho's Actor
on dfranzen's runtime match dfranzen's own Duck solver on levels? And does a per-decision call cap
of 4 help or hurt that Actor?

Posture: evaluation. Try to break the lean Actor at submission load, after pair-1 diagnosed its loss
as tokens per level rather than play quality.

Contrast (same weights, drafter, SGLang server, sampling, GPU, games and caps; each lean arm is the
v9 hybrid notebook with only its TYCHO_MODE/TYCHO_NOOA_TREATMENT line changed):
| Arm | Solver |
|---|---|
| control | dfranzen M2 notebook, unchanged Duck harness |
| lean_prune | Tycho Actor only (no_world_model, no Builder, no Scribe); closed decisions pruned (reasoning stripped; tool results, argument strings and prose bounded to 1,536 B; images dropped); newest grid inlined only; 64K window; lean system prompt; old ceiling of 25 tool-using calls per decision |
| lean_cap4 | lean_prune + at most 4 tool-using calls per decision, then one take_actions-only call (thinking off, 2,048 tokens, retried once); budget announced in the strategy block |
Contrasts: lean_cap4 vs control, lean_prune vs control, lean_cap4 vs lean_prune (the cap effect).

Prediction:
- Mechanism (lean arms). The mean Actor prompt_tokens is below 40K (pair-1: 63-103K). For lean_cap4, no decision exceeds 5 Actor calls and no game crashes from a missing action. For lean_prune, decisions longer than 5 calls do occur (ls20-like games).
- Tokens. Generated tokens per completed level for each lean arm are within 2x of control (pair-1 hybrid: 2-12x).
- Levels. Each lean arm totals within ±10 levels of control (INCONCLUSIVE), with lean_cap4 >= lean_prune. Historical controls on these games at 3600 s scored 30 and 28 levels.

Measure:
- Levels and score per game: <arm>/output/benchmark.json (control) or <arm>/output/tycho-dfranzen/benchmark.json, field game_runs[].levels_completed, final_score, actions_per_level.
- Tokens: the same file's history[].generated_tokens (control) or hybrid-games/*/compute.json completion_tokens (lean).
- Calls per decision and demand calls: hybrid-games/*/llm-trace.jsonl, grouped by sequence; demands are calls whose tool list is take_actions only, in the decision-checkpoints request.json.
- Prompt size: llm-trace prompt_tokens; projection window and evictions: context-projection.jsonl.
- Crashes and stop reasons: execution-receipt.json, benchmark game_runs[].state and solver_note.
Count a level only if completed at or before 3600 s of game wallclock.

Cap: 3 Kaggle sessions × 4,800 s hard cap (watchdog forced stop at 4,790 s, soft stop at 4,500 s), 1 RTX PRO 6000 each, 4.0 GPU-hours reserved in experiment/budget.json (16.78 unreserved before the reservation; Kaggle UI showed 26 h 16 m available on 2026-10-06). At most 2 concurrent sessions; the third launches when a slot frees. 10 games per arm, 10 concurrent, 3,600 s per game, one pass, no action cap. No money spend and no competition submission.

Stop:
- Each session stops at its caps. No arm is retried under this card.
- Launch order (seed 450281849): lean_cap4 and lean_prune together, then control when a slot frees.
- Do not launch control if either lean arm shows a crashed game, a GenerationError from a missing action, or a treatment that was not applied (Builder or Scribe calls in compute.json, or a projection window other than 65536). The lean arms would then be uninterpretable, and control spend would buy nothing.
- There is no futility stop mid-run: games are independent and the sessions are bounded.

Invalid if:
- Hardware other than one RTX PRO 6000, or changed weights, games or caps (pair.check fails, or pair-config.json differs).
- A third concurrent session, or a session that overlaps another account run and contaminates the quota delta.
- Missing benchmark.json or host-actions receipts, or unprovable actions.
- An infrastructure crash before gameplay.
Outcome and spend are kept either way; nothing is retried silently.

Result map (Δ = total levels over the 10 games; identical controls differ by about sd 5):
- lean_cap4 − control >= +10 (BETTER): replicate lean_cap4 vs control on the other 15 public games, then consider it as a submission candidate.
- lean_cap4 − control <= −10 (WORSE): diagnose from traces (calls per decision, deaths, tokens per level); do not iterate the arm blindly.
- Otherwise (INCONCLUSIVE): report Δ, the per-game sign count and levels per million tokens.
  - If a lean arm reaches at least control's levels per million tokens, run one replicate.
  - If not, close the Tycho-on-dfranzen line and work on dfranzen's harness directly.
- lean_cap4 − lean_prune >= +10: keep the cap. <= −10: drop the cap. Otherwise the cap's effect is under 10 levels; keep whichever arm has fewer tokens per level.
- Mechanism predictions that fail (lean prompts >= 40K mean, or crashes) take precedence: fix the mechanism before reading levels.

Outputs: solvers/tycho_dfranzen/experiment/wave-2/<arm>/{launch-response.json, status-history.jsonl, collection-receipt.json, output/}; full decision checkpoints remain on Kaggle (kernel output) and are fetchable via collect_outputs.KaggleOutput.

Not qualified before GPU: the real model's use of the cap and its prompt; the hardened per-game UID host at 10 concurrent games; Actor-only throughput at 10 streams; the cache effect of pruning once per decision; SGLang acceptance of the thinking-off action call.
