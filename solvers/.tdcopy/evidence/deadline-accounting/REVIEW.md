# Deadline accounting repair

The host wrote `compute.json` from its last IPC snapshot before retiring the worker. Completed calls inside an unfinished decision therefore disappeared from host and benchmark totals. q6 exhibited this in AR25 and SB26: eight responses / 17,634 completion tokens. Their raw provider receipts and durable trace agree.

The candidate now retires the worker first, reads its settled completed-response trace, reconciles calls, per-role completion tokens, prompt tokens and cached tokens, and credits only the remainder after already accounted actions. The execution receipt identifies the accounting source and the recovered difference. Non-token counters such as evictions still come from the last IPC snapshot. An incomplete final trace write or absent token usage remains explicitly unresolved; provider cancellation is never asserted. A missing trace preserves known IPC usage. Malformed committed records or a trace behind known IPC usage raise an integrity error.

The canonical source, frozen q6 source/outputs and pair configuration are unchanged. Only `src/hybrid/solver.py` and new `src/hybrid/accounting.py` change runtime behavior. The generated runtime/payload/notebook were rebuilt locally and verified; nothing was uploaded or launched.

## Plan and pre-mortem

Success: after a worker completes responses but never returns a decision, the final compute artifact and TAAF benchmark retain those responses without taking an extra action or counting earlier tokens twice.

Failure paths considered before the fix:

- Reading while the worker is still writing can miss or truncate the last response. Retire the process group before reading; treat an incomplete final JSONL record as unresolved.
- Adding all trace tokens to earlier action totals counts them twice. Reconcile the absolute completed total, then subtract the host's already credited total. Exercise both zero-action and one-action deadline cases.
- Counting an interrupted request as zero fabricates a precise budget. Preserve unresolved usage and cancellation uncertainty in the receipt.
- A missing/invalid trace could erase known IPC usage or strand a worker. Preserve the snapshot when no trace exists; reject committed corruption/regressing usage, and keep the existing cleanup/slot-release finally path.
- Bookkeeping could change solving behavior. The change runs after worker retirement; matched request/tool/action policy is untouched. Local real-GameAPI and role/prediction/repair fixtures exercise the rebuilt payload.

## Reproduction and checks

`tests/accounting_regressions.py` uses a real subprocess and protocol pipes, not a mocked worker. The pre-fix run failed twice: 31 uncredited tokens became 0, and 24 tokens after a seven-token action became 0. The repaired run retains 31 total tokens in both cases, performs respectively zero and one engine actions, and attributes 20 tokens to Actor / 11 to Builder. A partial final write stays unresolved. Replaying the preserved q6 traces produces 172/225,133 AR25 calls/tokens, 95/235,653 SB26, and 151/232,274 LS20 without editing their original artifacts.

Final commands, using `solvers/tycho_nooa/.venv/bin/python`:

- `tests/accounting_regressions.py`: six tests, including known q6 replay, missing/null usage, committed corruption and invalid usage/roles.
- `tests/qualify.py`: nine tests, including real offline AR25/LS20/SB26 GameAPI actions, live loopback role accounting and prospective prediction/repair, cancellation, slow HTTP deadlines and worker errors.
- `tests/review_regressions.py`: six tests, including process descendants, persistence failure cleanup and deadline sharing.
- `tests/launch_regressions.py`, `tests/pair_regressions.py`, `tests/collector_watchdog.py`: three tests each.
- `verify.py`: exact notebook/payload/runtime hashes, preserved donor cells and canonical provenance, package versions and notebook schema pass.

Logs reside alongside this report. Total: 30 passing local test methods. This verifies the local repaired artifact; it is not a new GPU execution or efficacy result.

## Risk measurement and review

Coverage.py and Radon are declared by the canonical solver but absent from this venv. No dependency or gate was added. `tests/accounting_risk.py` instead reports a bounded stdlib diagnostic: executed AST statement lines and an explicit AST decision count, with the CRAP formula and no threshold. It is not comparable to the canonical Radon/Coverage.py gate.

Using the same six-test `review_regressions` universe, `_play` remains at 68/117 covered statements. Its decision count changes 58 -> 59 and diagnostic CRAP 305.11 -> 314.70 because the receipt now marks incomplete/missing usage. This remains a high-risk host method. The six focused accounting tests measure the new helper separately; their final coverage and score are in `risk-accounting.json`. The new tests add behavioral coverage; they do not establish a comparable whole-host CRAP improvement.

The review inspected the source patch, trace producer, IPC snapshots, cancellation/teardown callers, TAAF final credit, build assembly, collector/audit expectations and the revised tests. No supported defect remains in this bounded fix. Interrupted provider work remains unmeasured, and a completed raw response can still precede a durable trace event at an abrupt kill; the receipt does not claim that all server work is measured.

`solver.patch` preserves the edit against the exact pre-fix candidate. The existing hybrid directory is untracked in the enclosing repository; the scoped commit records the new module/tests/report and this patch without absorbing the rest of the pre-existing hybrid work.
