# Research log: 0002 generic LLM baseline

## 2026-07-11 — baseline contract

Question: how much structure can a capable local multimodal LLM invent during
interaction when given public observations, transition history, and a generic
Python workspace, but no imposed ontology or planner?

The policy contract is intentionally small. The process measurement contract is
large but post hoc: retain full model/tool/action evidence, then annotate when a
compact representation appeared, whether it remained stable without new
evidence, and where actions were wasted.

No hybrid mechanisms are included. Color permutation is the first contamination
probe. Spatial/layout permutation remains blocked until a behavior-preserving
public action transformation is specified and tested.

## 2026-07-13 — first hosted Qwen smoke test

Question: can the generic controller complete at least one full
observe → inspect → decide → act → observe loop within a five-action,
fifteen-call, 50,000-reported-token budget?

The first run (`run-20260713T093753Z-bdd2c0ab`) returned zero actions and crossed
the token gate after four model calls (54,817 tokens). Python had returned
20,026, 14,013, and 3,849 characters in its first three calls; replaying those
outputs caused prompt usage to grow from 1,014 to 22,110 tokens per call.

The only mechanism changed was the generic model-visible Python-output bound:
20,000 characters became 4,000, with valid JSON truncation metadata. A final
self-describing rerun (`run-20260713T094854Z-d23e60c5`) recorded that bound in
its manifest. Qwen made nine model calls and eight Python calls, then issued
legal action 1 without repair. The action changed 52 pixels. Cumulative usage
reached 51,237 tokens, so the next decision stopped at the configured gate.

Conclusion: bounding Python output repairs context amplification and permits a
real environment transition, but it does not yet make the controller viable at
this budget. First-action latency remains eight tool calls. Raising the token
budget would conceal that control problem; the next experiment should isolate
why analysis does not terminate earlier.

## 2026-07-13 — experiment-first prompt probe

Question: does a prompt-only instruction to prioritize quick interaction over
initial-frame understanding reduce action latency under the same controller and
resource limits?

The `experiment-first` variant adds no mechanism or game knowledge. It tells the
model to wing it, use uncertainty as a reason to run legal probes, and reserve
Python for concise comparison and bookkeeping. The run
`run-20260713T100519Z-7e89c8a1` used the same model, game, seeds, five-action
limit, fifteen-call limit, 50,000-token gate, 4,000-character Python-output
bound, and disabled retrospective as the neutral run.

Qwen issued action 1 on its first model call and action 2 on its second, with no
Python calls. Both produced 52-pixel transitions. It then made six Python calls
to compare the two transitions before issuing action 3, which also changed 52
pixels. The run stopped before action 4 at 53,741 reported tokens. No level was
completed.

Compared with the neutral recorded run, first-action tool latency fell from
eight calls to zero and completed probes increased from one to three. This is a
single stochastic comparison, not a stable effect estimate. The trace does show
that prompt wording can redirect early control, while the remaining bottleneck
moves to post-transition analysis: failed dependency assumptions, schema
rediscovery, and verbose pixel-level comparison.
