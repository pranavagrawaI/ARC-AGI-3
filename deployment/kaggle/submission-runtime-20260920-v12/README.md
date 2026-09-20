# Transcript-eviction runtime v12

Successor to v11 (byte-identical to the `nooa-join/` that ran R8/R9). One change.

**The Actor transcript is bounded without a model call.** v11 shipped with NOOA's
summarizer disabled in `headless.py` — correctly, because that summarizer is a
background LLM request (`asyncio.create_task`) that can straddle the game
boundary and trip the teardown guard. But nothing then bounded the CodeAct
transcript. R9 cd82 (`analysis/actor-critic-construction-kaggle-r9-cd82-20260915T183130Z`)
measured ~5.5k prompt tokens per game turn — the full 64×64 grid in every task
prompt plus every cell output kept forever — 14.2k at turn 0, 73.6k at turn 10,
with the served 262k ceiling near turn 45 and a failed request (game over) past
it. Unreachable at 600 s and 11 actions; certain in a 200-action game.

`chassis/transcript_eviction.py` keeps the last `transcript_keep_turns` game
turns (default 4; env `ARC_TRANSCRIPT_KEEP_TURNS`, 0 restores v11) verbatim and,
before each model call, collapses everything older into one host-authored
digest rebuilt from `ipc/transitions.jsonl`: one line per executed action
(turn, step, action, changed cells, level/terminal marks), capped at 2k chars.
No LLM request is scheduled. Continuity lives where it already lived — the
Actor's notes and helper module, and `trajectory()`/`last_transition()`/
`observations()` — not in the raw transcript. Evictions are recorded in
`ipc/transcript_evictions.jsonl`. This is the whole-block eviction Duck's
harness uses (`tool_agent.py` `_drop_oldest_history_block`), minus the model-
authored notebook, which Duck's own memory audit found absent in half its
episodes with no effect on levels.

Measured with the scripted-LLM fixture (`test_transcript_eviction.py`, keep=3):
prompt 49k → 72k chars over the first three turns, then 62k ± 0.3k for the rest.

Not changed: budgets, critic, prompts, tool surface, worker protocol, launcher.
`build_notebook.py` ships the new module; `test_every_frozen_chassis_file_ships_in_the_notebook`
keeps the freeze list and the ship list from diverging again.

Qualification: local deterministic only (180 tests). No loopback or Kaggle run.
`LAUNCH.json` is `not_authorized`. The carried `kernel-metadata.json` still names
the v10 kernel and must be re-slugged before any push
(see kaggle-kernel-title-must-slugify-to-id).

---

# Objection-closure runtime v10

Successor to v9 after the 2026-09-12 objection-closure public run
(`analysis/deployment-probes/20260912-objection-closure-public-v1`), which ended
in ERROR on LS20 with two games unplayed. The Actor declared the game "stuck"
with 152s and 193 actions remaining and returned DONE; the worker fell silent and
the host's decision timeout killed the launcher.

Two changes, both removals.

**The Actor no longer owns "game over".** Its instructions no longer tell it to
call `return_result(RespondReason.DONE)` on a "stuck" game; every turn ends with
`submit_actions`. If it calls DONE anyway on a NOT_FINISHED game, `headless.py`
records the refusal in `done_refused.jsonl` and redelivers the same state with a
host note. No new game-end mechanism is added: if the model budget really is
spent, the existing admission gate ends the game through `budget_exhausted.json`.

**The critic packet no longer carries a projected action count.** In the public
run the critic restated `affordable_actions_at_this_rate: 2` as "only two
wall-clock actions remaining" and the Actor adopted it as a rule. The packet
keeps wall seconds remaining and seconds per executed action — facts — and
states that the host ends games.

Everything else is v9: compact commitment packet (three code items, five
commitment rationales, measured transitions as counts and regions, the objection
ledger), one pending objection at a time, dispositions on commitments,
recurrence and probe accounting, expiry after three refused commitments.

On the one game v9 ran, the contract behaved as specified and the critic cost
1.3% of session wall (v8: 26.0%) at 1.7-2.2k prompt tokens per call.

This candidate has local deterministic and loopback qualification only.
`test_done_refusal.py` reproduces the public-run failure on v9 (`worker timed
out waiting for decision`) and passes on v10.
