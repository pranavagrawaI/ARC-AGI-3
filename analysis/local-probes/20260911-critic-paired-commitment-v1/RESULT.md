# Paired commitment diagnostic: no demonstrated incremental repair

The contract, prompts, budgets and evaluator-only rubric were frozen before three calls. All three completed on the existing Qwen3.8-27B CoreWeave FP8 route, with zero reasoning tokens. The critic returned no_concern. Both Actor continuations proposed one LEFT action. Neither explicitly repaired the selected commitment.

| Output | Observed behavior | Time | Completion tokens |
|---|---|---:|---:|
| Unreviewed Actor | Discussed a possible premature log entry and uncertainty; proposed one LEFT probe | 5.81s | 301 |
| Critic | Approved bounded exploration and the marking of movement mechanics as untested; did not discuss the selected target-knowledge claim | 4.48s | 85 |
| Reviewed Actor | Printed incorporate and retained the LEFT probe; did not narrow the selected claim | 4.82s | 243 |

Total cost: **$0.0161358**; elapsed transport/run time **15.11s**. The reviewed branch's critic plus Actor calls took about 9.30s. These are individual observed request times, not a performance estimate. Both Actor requests reported 17,248 cached tokens; the independently instructed critic reported zero.

## What was actually compared

The source was v4 skill-arm request17, immediately after the Actor wrote a hypotheses note in cell16. Its selected commitment included:

> no alignment hypothesis is testable without knowing target.

The critic received the complete captured request and the complete source cell, without evaluator diagnosis or later/control output. Both Actor requests retained the exact same captured message context; only the reviewed arm gained the critic's output and frozen response instruction. Their generation settings and 1,024-token ceilings matched. The critic had its own system instruction, no tools and a 384-token ceiling covering all generated completion tokens. The Actor's reviewed acknowledgement had a 60-word target; it emitted 79 words, so concision was not satisfied despite staying within the hard token budget.

This was a **proposed-continuation comparison**. The in-process Python state was not checkpointed, so no generated cells were executed. Proposed note changes did not become persisted repairs and proposed LEFT actions did not produce environment observations. The source context already included the prior epistemic self-review skill in both arms; this tests incremental review conditional on that history, not a pristine baseline.

## Why the result is inconclusive on detection

The separately frozen rubric identified a candidate gap: uncertainty about the true target need not prevent defining provisional candidates and investigating them. It also preserved the legitimate alternative reading: an unspecified alignment idea needs a specified candidate before it has concrete predictions. Criticism must distinguish these readings rather than automatically declaring the sentence wrong.

The critic addressed the bounded movement probe and untested mechanics elsewhere in the source cell. It did not explain how it interpreted the alignment statement. Its no_concern answer therefore does not demonstrate detection of the candidate gap, but the precommitted ambiguity prevents a clean detection-failure verdict. Nor does the review establish that every commitment in the material was warranted.

The unreviewed Actor independently discussed a potential source-recording issue and continued with a bounded probe. That does not demonstrate independent repair of the selected target-knowledge claim. The reviewed Actor acknowledged approval of the probe rather than changing that claim. Silence on the target is not counted as either persistence or correction.

**No incremental epistemic repair was demonstrated.** There was also no established harmful adoption. A different action was not required for success; an explicit warranted narrowing of the selected belief with the same action could have earned local proposed-repair credit.

## Retained decision

The interaction protocol was mechanically exercised, with execution statuses separate from assessments, hidden grading, complete shared context, source metadata and bounded output. It did not earn another architectural layer or an efficacy claim. A future diagnostic should select an unambiguous commitment with clear decision consequences and identify that commitment neutrally, while retaining the complete context. A broad source cell can let the critic approve a different, reasonable part of the Actor's thinking.

No further calls were made under this card. Four focused preflight tests passed. Frozen inputs remained unchanged after inference. No gameplay, Kaggle run, deployed changes or generated-code execution occurred.

## Evidence and reproduction

- `CONTRACT.md`, `EXPERIMENT-CARD.md`, `GRADING.json`, `FROZEN.json`: pre-inference contract and interpretation.
- `00-control`, `01-critic`, `02-reviewed` request/response/timing JSON files: exact paired inputs and outputs.
- `BINDING.json`, `REVIEW.json`: host source identity and model judgment kept separate. No quotation was supplied for no_concern, so quote binding is not applicable.
- `GRADES.json`: explicit semantic grading against the frozen rubric.
- `AUDIT.json`, `ACTOR-PROPOSALS.json`: reproducible mechanical checks and emitted cells, with no code execution.

Recompute with `deployment/kaggle/.venv/bin/python experiments/probes/epistemic-critic-local/audit_paired_commitment.py analysis/local-probes/20260911-critic-paired-commitment-v1`. The auditor parses Python syntax to extract literal proposals; it does not run them.
