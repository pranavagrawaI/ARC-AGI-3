## 2026-10-05 execution update

Pair-1 is frozen and both private v1 arms are accepted/queued. New candidate-only bounded Builder lifecycle and interrupted-usage receipt fix are locally qualified (40 checks). See [pair-1/STATUS.md](pair-1/STATUS.md), [card](pair-1/EXPERIMENT-CARD.md) and [review](../evidence/builder-lifecycle/REVIEW.md). This dated update supersedes older statements that the Builder policy is unchanged or the comparison set is provisional. No efficacy result yet.

# Hybrid mechanism experiment

Authority: Pranav confirmed the completed interview on 2026-10-03. Total cap: 10 GPU-hours across the next 2–3 calendar days, including failed runs, startup and qualification. No competition submission or public publication. Local reversible implementation and qualification are authorized.

Question: At equal accelerator allocation and elapsed compute time, does the hybrid improve official score, and which reasoning mechanisms earn their cost?

The whole-system contrast is unchanged dfranzen versus dfranzen infrastructure with Tycho/Nooa Actor-controlled Builder, executable world model and Scribe. This compares bundles; it cannot attribute a difference to reasoning alone. Attribution requires a subsequent matched intervention on a component, with the other components fixed.

## Sequence and resource reservation

Updated after Pranav's 2026-10-04 clarification and accepted recommendation: SAME model, SMALL GAME SET, Actor-controlled construction, code review first, and at most 1–2 GPU-hours for further qualification. Voluntary Builder activation is not a performance-run pass condition.

1. Local review and qualification: reproduce supported defects, preserve the old candidate, verify real worker/runner execution, construction/repair, prospective predictions, action continuity, deadlines, cleanup and provenance. No paid endpoints or GPU.
2. Prior gpu-q4 smoke: completed, conservatively charged 1 GPU-hour. Nine remain. It demonstrated execution but rejected SB26's requested Builder call; it is not construction efficacy evidence.
3. Further qualification: plan one 1-GPU-hour run, within the user's 1–2-hour allowance. Fixed ar25/ls20/sb26, one pass, three concurrent games, 2400 seconds/game, no action-count cap, same FlashNext weights/serving. Freeze a separate card before launch. A justified second qualification hour uses contingency and a distinct card. v3 allocates Builder time when requested; the Actor remains free not to request it.
4. Whole-system pair: provisionally protect 3 GPU-hours, 1.5 per arm including setup, on the same three games. Freeze identical hardware, model, serving, game versions/order and time/action caps. This compares bundles, not pure reasoning. Inspect receipts before follow-up.
5. Conditional mechanism pair: protect 3 more GPU-hours, 1.5 per arm. First target is executable model construction/repair. Choose the precise intervention from discovery traces, freeze it before fresh matched runs, and distinguish Builder delegation, canonical model support and feedback. If unused, skip an inert ablation: choose justified depth/replication or leave budget unspent. Non-use cannot establish whether construction would help if used.
6. Contingency: 2 GPU-hours. Do not consume the mechanism-pair reservation for further whole-system exploration without recording an explicit revised decision. Each retry is a new card/run. Insufficient evidence remains inconclusive; the cap does not require a positive or negative verdict.

This envelope totals 10 hours including q4; budget.json records actual reservations. Reserve BOTH arms before starting a pair. The 2–3-day calendar limit includes Kaggle queue delays.

GPU-hour accounting is accelerator count times allocated elapsed hours, not sum of concurrent game time or model latency. Until a run ends, reserve its entire hard timeout. Each launch must fit cumulative observed consumption plus outstanding reservations within 10 hours. Kaggle hard timeout is required; also retain graceful in-notebook shutdown and partial artifacts. Do not launch another copy while status is unresolved.

## Measurements and interpretation

Primary: official benchmark score at matched resource caps, supported by level completion/action receipts. Report paired game deltas, failures and unfinished games; small single pairs remain provisional.

First-shot mechanism correctness: first recorded testable prediction about a game mechanism, committed before the corresponding observation. A correct explanation written after observing a transition does not count. Untestable or absent claims remain unmeasured, not correct.

Recovery: elapsed compute time and model/engine actions from the first observed contradiction to the first subsequently verified corrected prediction. Cases without recovery are censored at the cap. Preserve prediction source/model hash and raw observations so the scoring can be inspected.

Mechanism metrics explain outcomes; they do not override score. Better predictions or faster recovery with lower score identify a promising direction, not a better solver. Scripted fixtures establish execution behavior, not model capability.

Invalidity: source/config drift, wrong model or game list, mismatched resource caps, unaccounted execution after cutoff, dropped/duplicated action acknowledgements, absent raw evidence, or unintended mechanism suppression. Classify operational validity before interpreting efficacy. A valid zero is a result; missing output is pending.

Old candidates/evidence are preserved under revisions/local-v1 and revisions/local-v2-reviewed. v2 changed trigger -> orchestrator. v3 changes invocation-time Builder allocation, fixes IPC/cleanup defects and distinguishes game-budget expiry from early decision failure. These are new candidate changes, not retroactive repairs of q4. See REVIEW.md and PRE-MORTEM.md. GPU q5 failed before gameplay on the GameAPI game-name fallback. The user explicitly authorized relaunch with fixes; source-v4 was uploaded, CPU q6 passed the packaged real-GameAPI rehearsal and receipt audit, and the distinct gpu-q6 was accepted with a one-hour reservation from contingency. Seven GPU-hours remain unreserved, protecting the two planned three-hour pairs and one contingency hour. GPU q6 completed: mean score 30.23, 437 actions, two admitted Builder attempts both timed out and rolled back. Host accounting missed eight completed final-decision responses; raw receipts reconcile the 17,634-token difference. See gpu-q6/RESULT.md and AUDIT.json; The editable candidate now repairs deadline accounting and passes local tests/replay; see evidence/deadline-accounting/REVIEW.md. Builder behavior is examined in gpu-q6/BEHAVIOR.md; its reasoning policy is unchanged. The approved notebook-log elapsed fallback charges q6 about 0.83734 hours, leaving about 7.16266 unreserved hours; earlier charges remain unchanged. The later comparison set is provisional in COMPARISON-DRAFT.md and is not frozen.

## Run cards

Freeze one concrete card per GPU launch with question, contrast, prediction, metrics, exact hashes, numeric cap, stop condition, invalidity conditions, result map and output paths. This plan is not a frozen run card. GPU launches remain gated on passing local qualification; the user's budget authorization does not waive execution validity.

## Pair decisions — Pranav, 2026-10-04 (supersede conflicting text above and in COMPARISON-DRAFT.md)

Grounds: PRE-MORTEM-REVIEW-20261004.md and the grilling on board thread 504 (#530).

- Gate: the whole-system pair launches only if gpu-q6 is operationally valid (no crashed game, receipts complete) AND the Actor requested and the host admitted at least one Builder call. Without Builder use, contingency buys a qualification run that diagnoses non-request, not the pair.
- Games: ls20/re86/sc25, frozen; all three count toward the verdict.
- Margin per game: max(observed donor run-to-run spread, score value of one level on that game).
- Result map: WIN = hybrid beats control by the margin on >=2/3 games -> run the mechanism pair. LOSS = hybrid worse by the margin on >=2/3 -> stop GPU spend, diagnose offline from traces. INCONCLUSIVE -> run the mechanism pair only if the Builder chain reached a checked prospective prediction in >=2 games; otherwise release the reservation. Runs are noisy, so the per-game chain report (offered/requested/admitted/completed Builder calls, prospective predictions, correctness, recovery) is produced for every verdict; it explains why, it does not override score.
- Charging: Pranav reads remaining account GPU quota immediately before and after the pair; the delta is charged to both arms together. If a reading is missing, charge notebook-log-proven elapsed time. Other sessions overlapping the window contaminate the delta; record them.
- Launch: both arms concurrently in the two GPU slots, back to back after q6 completes; post a board hold first; launch order from the recorded draw in pair-identity.json.
- Budgets are not re-derived per arm: src/hybrid/pair.py (3600 s/game, concurrency 3, one pass, no action cap, session 5400 s, soft stop 5100 s) is embedded verbatim in both notebooks by freeze_pair.py, and the hybrid runner's compare mode only checks it. The donor's cell-16 formula would otherwise allot 870 s/game at three games. tests/pair_regressions.py covers this.
