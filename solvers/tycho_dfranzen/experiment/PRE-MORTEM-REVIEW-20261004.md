# Pre-mortem, reviewer pass — 2026-10-04 01:55 IST (claude-tdf-review)

This adds to `PRE-MORTEM.md` (Codex, 01:47) and doesn't replace it. That file covers the experiment design well: mechanism attribution, silent non-use, proxy correctness, unequal work, three-game luck, budget drain, evidence collection and stale qualification. This pass looks only for the paths it misses or underrates. Nothing was launched or edited apart from this file.

Plan under test: v3 qualification, then a small-set whole-system pair (ar25/ls20/sb26, 1 pass, 3 concurrent, no action cap, `COMPARISON-DRAFT.md`: 5400 s per arm, soft stop 5100 s, 3600 s/game), then a conditional mechanism pair. Budget: 9 of 10 GPU-hours remain, with each launch charged at its full cap.

**Failure scene, 2026-10-06:** the budget is spent. The pair reports donor 16 levels and hybrid 13 on three games. One hybrid game shows `crashed` at minute 41, and ar25/sb26 are 8/8 in the donor arm. Nobody can say whether Tycho's reasoning helped, hurt, or never got a chance. The mechanism pair didn't fit in what was left.

## P1. The game set cannot discriminate (decision-reversing)

- **Chain:** the smoke games were reused as the comparison set → the donor saturates two of the three within the cap → the hybrid's upside on those games is close to zero → only ls20 varies, at about 1–2 levels, inside run-to-run noise → the pair can only show "hybrid worse" or "inconclusive".
- **Evidence (observed):** donor levels by minute 60 of the game clock, which includes about 16–30 min of startup. Values from `analysis/m2-winners-20261001/runs/*/benchmark.json`, runs control / control-r2 / s1:

  | game | n | @60 min | final |
  |---|---|---|---|
  | ar25 | 8 | 8 / 7 / 5 | 8 / 8 / 8 |
  | sb26 | 8 | 7 / 8 / 8 | 8 / 8 / 8 |
  | ls20 | 7 | 1 / 1 / 2 | 2 / 2 / 4 |

  Those runs had 25 games behind a gate of 10. A three-game arm gets more GPU per game, so it should saturate sooner.
- **Rating:** likelihood high, impact high (3 GPU-h buy no answer), detectable now, recoverable now and not after launch.
- **Earliest signal:** the table above.
- **Response (prevent):** choose games where the donor sits in the middle of its range at the cap in all three runs. Candidates by minute 60: re86 (5/5/3 of 8), vc33 (4/3/3 of 7), sc25 (4/4/3 of 6), r11l (2/4/4 of 6), dc22 (2/2/1 of 6). Keep ls20 if wanted. This changes the frozen game list, so it's Pranav's call. Per-game variance stays large (s5i5 1/5/3, cd82 5/0/3), so state in advance which delta would change the decision.

## P2. Card, code and control config drift apart

- **Chain:** the card says one thing, the code does another, and editing the donor notebook silently changes its budgets → the arms aren't matched → the score delta measures configuration, not reasoning.
- **Evidence (observed):**
  - Hybrid `small` mode hard-codes 2400 s/game (`src/hybrid/runner.py:49`); the draft card says 3600 s/game.
  - Donor cell 17 recomputes `max_runtime_s_per_game = 532*60*concurrency//110` under priority scheduling. Setting concurrency to 3 for the control gives **870 s/game** without any warning.
  - The donor selects games by exclusion (`demo_excluded_games`), so a typo silently adds or drops a game. The hybrid asserts its exact set.
  - The draft's 5100 s soft stop isn't implemented. Cell 21 derives `soft_end` from `target.max_runtime_s` = 32400.
- **Rating:** likelihood high unless checked, impact high, poorly detectable from scores, cheap to fix before launch.
- **Earliest signal:** the two arms' `candidate-config.json` / solver repr differ in seconds, concurrency, game names or soft_end.
- **Response (prevent and detect):** set both arms' budgets explicitly after cell 17 and assert equality, including an exact inclusion assertion in the donor. Diff the two frozen notebooks cell by cell and allow only the solver swap. Make every number in the card match the code.

## P3. A runaway session eats the remaining budget

- **Chain:** any hang outside the per-game deadlines (runner shutdown, server startup, an unbounded cleanup wait) → nothing in the notebook enforces the cap (q4's "external 3600-second hard timeout" is only a value written to `qualification-start.json`, and `collect_run.py` never stops anything) → the session runs to Kaggle's maximum → most of the 9 GPU-hours are gone.
- **Evidence:** the missing enforcement is observed. The hang itself is inferred: the known hang paths are fixed, and the unknown ones are by definition untested. Kaggle has no CLI cancel, so recovery needs a human in the UI.
- **Rating:** likelihood low, impact catastrophic, detectable only by someone watching, recovery manual.
- **Response (prevent):** a watchdog thread in the first cell. At cap minus margin it sends SIGTERM to the runner, waits briefly, then kills the server and the kernel process. Also set `soft_end = min(card soft stop, donor)`. Put the same watchdog in the donor arm.

## P4. One fault takes out a third of an arm, or the whole arm

- **Chain A (game-level):** v3 removes the 16-action cap and enables the Builder, so long Actor tool loops now actually happen → one decision exceeds the 116K projection budget → `ValueError` is not recognised by NOOA's context-window recovery → worker error → that game is `crashed` for the rest of its time.
- **Chain B (arm-level):** an exception from teardown leaves `_play` (artifact write, close) → `_run_games` stops every sibling game, which Codex's REVIEW.md describes as intended draining → the whole arm ends early.
- **Evidence:** both mechanisms are observed in local reproductions (board 504#521). How often they happen live is unknown. q4 never exercised them (about 1 Actor call per decision, 16 actions).
- **Rating:** chain A medium likelihood, chain B low; impact high, because with 3 games one crash is a third of the arm and is easily misread as weak reasoning. Detectable with the right trace.
- **Earliest signal:** in `context-projection.jsonl`, `estimated_input_tokens` climbing toward 116K within a single sequence; `stop_reason=error` in any `execution-receipt.json`.
- **Response:** prevent A by eliding older reasoning and tool bodies inside the newest exchange, and declare any change to the 96 KB tool cap as a hybrid policy change. Contain B by catching `Exception` per game and stopping everything only on cancellation. Interpretation rule, frozen in the card: a crashed game makes the pair operationally invalid, not a zero.

## P5. Ledger arithmetic turns cheap failures into lost experiments

- **Chain:** each launch is charged its full cap → a packaging failure at cell 9 (wrong source-v3 hash or dataset, which is likely while the tree is still moving) costs minutes of real GPU time but 1–1.5 h on the ledger → two such failures use up the 2 h contingency → the mechanism pair no longer fits.
- **Evidence:** the accounting rule is observed (`budget.json`: q4 ran about 20 min and was charged 1 h). The source-v3 dataset isn't uploaded yet, and the tree changed several times on 10-04 between 01:27 and 01:43.
- **Response:**
  - Prevent: run cells 0–11 (install, payload and wheel hashes) as a CPU-only Kaggle session before any GPU launch. CPU sessions don't count against the GPU session cap. Whether that docker image/machine shape starts without a GPU is unknown; a failed CPU attempt costs nothing.
  - Decision for Pranav: when the notebook log proves elapsed time (q4: about 20 min), may that measured time be charged instead of the full cap? This choice decides whether a mechanism pair stays affordable.

## P6. Shared account capacity splits or blocks the pair

- **Chain:** other agents (m2_basics, m2_memory, m2-winners) share the 2-session GPU cap and the 30 h weekly quota → the second arm queues for hours or push hard-fails at quota → the arms run under different conditions or the pair can't finish within 2–3 days.
- **Evidence:** the caps and the 2026-09-25 quota exhaustion are observed (memory). Current usage is unknown. The quota reset Sat 2026-10-03 00:00 UTC.
- **Response (detect):** check quota and active sessions immediately before reserving the pair. Launch both arms back to back, and announce the reservation on the board.

## Examined, out of scope, accepted

- **Examined:** PLAN.md, COMPARISON-DRAFT.md, PRE-MORTEM.md, REVIEW.md, budget.json, q4 card/notebook/collector, current runner/solver/transport/policy, donor cells 4/17/21, three donor benchmark runs.
- **Not examined:** SGLang behaviour under the hybrid's request mix (images, three roles, prefix churn), server-side cancellation, the not-yet-written mechanism-ablation diff.
- **Changes worth making now:**
  1. Re-pick the game set (P1, Pranav's decision).
  2. Align the card with the code and lock the donor budgets with assertions (P2).
  3. Add the watchdog and soft stop to both arms (P3).
  4. Apply the P4 fixes plus the crash-invalidates-pair rule.
  5. CPU dry run of the install cells (P5).
- **Accepted:** an n=1 pair is a screen, not an estimate. Provider-side compute after cancellation stays unmeasured. Nothing here generalises to public-25 or the hidden set.
