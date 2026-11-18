# Hybrid mechanism experiment

Authority: Pranav confirmed the completed interview on 2026-10-03. Total cap: 10 GPU-hours across the next 2–3 calendar days, including failed runs, startup and qualification. No competition submission or public publication. Local reversible implementation and qualification are authorized.

Question: At equal accelerator allocation and elapsed compute time, does the hybrid improve official score, and which reasoning mechanisms earn their cost?

The whole-system contrast is unchanged dfranzen versus dfranzen infrastructure with Tycho/Nooa Actor-controlled Builder, executable world model and Scribe. This compares bundles; it cannot attribute a difference to reasoning alone. Attribution requires a subsequent matched intervention on a component, with the other components fixed.

## Sequence and resource reservation

1. Local execution qualification: no paid endpoint or GPU. Prove role delegation, executable model edits, prospective predictions, observed contradiction, repair, action/observation continuity, timeout containment, request shape, and packaging provenance.
2. GPU qualification: reserve at most 1 GPU-hour including server startup. Offline installation, hardened worker, real request/tool/replay and three-game smoke. Compatibility evidence only.
3. Matched whole-system pair: reserve at most 4 GPU-hours total, 2 per arm including setup. Same hardware, model, serving, game list, passes, action caps and per-game/wall limits. Preserve failed/unfinished games and exact resource accounting. Do not substitute an old donor score.
4. Mechanism contrast: reserve at most 4 GPU-hours total, 2 per arm. Select one mechanism from the qualified whole-run traces before freezing its card. Compare hybrid against hybrid with that mechanism removed or altered. Score and resource changes determine whether it earns its cost.
5. Unallocated contingency: 1 GPU-hour. A retry is a new card/run ID and consumes this cap; never silently retry. If startup or failures make the remaining contrast unaffordable, report the incomplete conclusion rather than overspend.

GPU-hour accounting is accelerator count times allocated elapsed hours, not sum of concurrent game time or model latency. Until a run ends, reserve its entire hard timeout. Each launch must fit cumulative observed consumption plus outstanding reservations within 10 hours. Kaggle hard timeout is required; also retain graceful in-notebook shutdown and partial artifacts. Do not launch another copy while status is unresolved.

## Measurements and interpretation

Primary: official benchmark score at matched resource caps, supported by level completion/action receipts. Report paired game deltas, failures and unfinished games; small single pairs remain provisional.

First-shot mechanism correctness: first recorded testable prediction about a game mechanism, committed before the corresponding observation. A correct explanation written after observing a transition does not count. Untestable or absent claims remain unmeasured, not correct.

Recovery: elapsed compute time and model/engine actions from the first observed contradiction to the first subsequently verified corrected prediction. Cases without recovery are censored at the cap. Preserve prediction source/model hash and raw observations so the scoring can be inspected.

Mechanism metrics explain outcomes; they do not override score. Better predictions or faster recovery with lower score identify a promising direction, not a better solver. Scripted fixtures establish execution behavior, not model capability.

Invalidity: source/config drift, wrong model or game list, mismatched resource caps, unaccounted execution after cutoff, dropped/duplicated action acknowledgements, absent raw evidence, or unintended mechanism suppression. Classify operational validity before interpreting efficacy. A valid zero is a result; missing output is pending.

The existing local-v1 candidate and evidence are preserved under revisions/local-v1. The v2 policy correction is trigger -> orchestrator. Actor reserve can suppress Builder under short timeouts; record actual decision budgets and Builder rejection traces.

## Run cards

Freeze one concrete card per GPU launch with question, contrast, prediction, metrics, exact hashes, numeric cap, stop condition, invalidity conditions, result map and output paths. This plan is not a frozen run card. GPU launches remain gated on passing local qualification; the user's budget authorization does not waive execution validity.
