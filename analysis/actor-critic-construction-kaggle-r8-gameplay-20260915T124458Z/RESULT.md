# R8 gameplay result — actor-critic-construction-kaggle-r8-gameplay-20260915T124458Z

Machine-readable: `R8-RESULT.json`. Raw receipts: `runtime-evidence-exact-request/` (22 files, sha256 in the JSON),
`push-*.json`, `quota-*.json`, `target-list-*.json`, `r8-current-status-*.json` (78 polls), `R8-CURRENT-TERMINAL.json`.

## Validity (checked before any number was read)

| Check | Result |
|---|---|
| Arm equality (`ARM-DIFF.json`, computed remotely before play) | `equal: true`; differing keys exactly {ARC_CONSTRUCTION_ARM, NOOA_AGENT_COMMAND, RECORDINGS_DIR, TYCHO_KAGGLE_WORKSPACE_ROOT}; forbidden differences [] |
| Both arms received the full 600 s admission | A 607.1 s, B 603.8 s; both ended `model_admission_budget` |
| Official recording ↔ driver progress | reconciled in both arms (recorded frames = executed actions; final state/levels agree); `game-progress.jsonl` == `games` in the launcher result |
| Construction in the baseline | none (arm A: no freeze, no request, 0 constructor calls) |
| Actor DONE / transport failure / cap stop | none (no errors, rejections, admission-stops or cleanup files produced) |
| Accounting | 132 admitted = 132 receipted, complete; server 133 = 132 + 1 warmup, whole-model count verified; purse 258 not approached |
| Teardown | `PROCESS-GUARD.json` passed, `killed_remaining_descendants []`; `SHUTDOWN-REAP.json` retired with **zero** leftovers; orderly shutdown passed; GPU processes after: none |
| Provider | COMPLETE (current-identity status; exact `version_label=1` status 404 as before; all files by `version_number=1`) |

**Verdict: VALID.** Finding 7's guard did not fire; finding 6's slack was 213 s (arm B started with 1128 s remaining, 915 required).

## Observation

| | Arm A (baseline, construction off) | Arm B (construction overlay) |
|---|---|---|
| Executed actions (official recording) | **10** | **19** |
| Levels completed / state | 0 / NOT_FINISHED | 0 / NOT_FINISHED |
| Actor calls | 57 | 60 |
| Actor calls per executed action (P5) | **5.7** | **3.16** |
| Critic calls (wall share of admission) | 6 (1.5%) | 8 (2.1%) |
| Constructor calls | 0 | **1** — `submit_candidate` directly, 14.5 s, 19,439 prompt tokens → digest-valid freeze |
| Freeze lifecycle | — | requested → frozen → `query_frozen_predict` → `assimilate_frozen`, bound to an executed (action, step) |
| Actor wall inside admission | 584 s of 607 | 558 s of 604 |
| Actor prompt tokens, mean / max | 51,906 / 90,894 | 55,636 / 93,944 |
| `submit_actions` calls in Actor output vs executed | 18 vs 10 | 24 vs 19 |

References (single runs, same game, official loop): v10 LS20 6 actions / 13.3 calls per action; v8 LS20 17 actions / 3.7.

Timeline: push 15:11:56Z → running 15:12:01Z (no queue) → supervisor 15:12:35–15:39:17 (1602 s) → arm A frames
15:20:07–15:29:02 → arm B 15:29:32–15:39:06 → COMPLETE first observed 15:51:43Z. GPU charged 0.45 h (2.27 → 2.72).

## Predictions

- **P1 activation — held.** Construction fired after the first executed transition and froze on its first request,
  within the 2-request cap.
- **P2 use — held.** Query and assimilate both completed and are digest-bound to executed steps.
- **P3 cost — falsified.** The card predicted B ≤ A executed actions (construction spends ≥ 45 s and adds prompt
  text). Observed B 19 > A 10. The constructor cost 14.5 s, not 45. This is the case the card flagged as "a
  surprise worth investigating", and it is one pair.
- **P4 levels — held (0/0).** No level in either arm; nothing to compare.
- **P5 tempo — reported.** 5.7 and 3.16 calls per action against v10's 13.3; both far below v10 on the same game and
  model, and the baseline arm ran the unmodified v11 worker, so the tempo change relative to v10 belongs to the
  chassis/runtime differences accumulated since v10, not to construction.

## Interpretation

1. The construction join runs under the real official `Agent.main` loop at gameplay scale and its lifecycle
   completes with one model call. That was the machinery question; it is answered.
2. Arm B executed roughly twice the actions of arm A with the same Actor, model, game and budget. With n = 1 and
   LS20 Actor variance unknown (v8 17, v10 6, here 10 and 19), this is an observation, not an effect. What can be
   said: construction did not cost tempo in this pair, and the arm with the freeze spent fewer Actor calls per
   action (3.16 vs 5.7). Whether the overlay's prompt/lifecycle text changes the Actor's behaviour independently
   of the freeze is not separable here.
3. Neither arm made progress toward a level in 600 s. The Actor is spending ~10 s per call on ~52k-token prompts;
   ~95% of admission is model wall. Levels are not within reach of this envelope with this Actor.
4. Submitted-vs-executed gaps (18→10, 24→19) mean the host declined or collapsed some submissions; attribution
   needs `ipc/actions.jsonl`, which was not retrievable (see gaps).

## Evidence gaps (honest)

- Per-arm `ipc/*.jsonl`, `host-events.jsonl`, `FROZEN.json`, constructor `RESULT.json` were not retrieved: the
  workspace session directory names carry a random suffix the driver did not record, and tree listing is the
  known 429 trap. Lifecycle facts are the booleans `classify()` computed remotely over each arm's workspace.
  Follow-on fix: record each arm's session directory in the driver summary.
- `requests.jsonl` / `logical-requests.jsonl` exceed the 16 MB retrieval bound; receipts (raw-body sha256, usage)
  are in the launcher result.
- Official recorder `action_input` is `{id: 0, data: {}}` on every OFFLINE frame; action identity comes from the
  Actor's `submit_actions` calls and the executed-count reconciliation.

## Permitted next action (per the card's result map: valid, B ≥ A, P1+P2 held)

- Freeze this identity; no further pushes.
- The card's branch: repeat on S5I5 and CD82 in a new identity before any "better" wording. Prerequisites:
  stage the four exact public files (`environment_files/s5i5/18d95033/{s5i5.py,metadata.json}`,
  `environment_files/cd82/fb555c5d/{cd82.py,metadata.json}`) via `kaggle competitions download -f …`, hash them
  against public manifest `842477b6…`, add pinned entries to `game_preflight.py`, extend `GAMES`, and have the
  driver record session directory names. Same envelope fits two arms × one game per identity, or one arm × two
  games — not two arms × three games in one hour.
- No competition submission is authorized or implied by this result.
