# Result — demonstration treatments did not improve efficiency

## Verdict

C and D completed two real actions cleanly, but both failed the efficiency prediction. C required 18 physical requests and 170.41 seconds. D required 29 requests and 469.87 seconds. The final A control consumed the cumulative reservation cap after 25 requests and zero actions, so this run cannot provide a clean same-run treatment effect. Combined with the prior valid A replicate (11 requests, 88.18 seconds), neither C nor D is a promotion candidate.

## Measurements

| Arm | Outcome | Actor / summarizer requests | Wall | Prompt / cached tokens | Completion tokens | Cost |
|---|---|---:|---:|---:|---:|---:|
| C: demonstration | 2 actions, 0 levels | 15 / 3 | 170.41 s | 309,964 / 181,888 | 6,074 | $0.0967356 |
| D: demo + evidence | 2 actions, 0 levels | 22 / 7 physical | 469.87 s | 521,976 / 228,928 | 20,678 | $0.2135924 |
| A: baseline | 0 actions; reservation stop | 23 / 2 | 236.81 s | 423,986 / 279,888 | 10,844 | $0.1321544 |

D's last summarizer completed upstream during cleanup, so its 29 physical responses reconcile to 28 worker records plus one completed-not-delivered response. All reported reasoning-token counts were zero. No deep investigation was invoked. Total reported provider cost was $0.4424824; no forwarding retry or relaunch occurred.

## What the agents did

C made 14 Actor calls before its first action (`LEFT`) and three more before its second (`RIGHT`). It wrote no decision-experiment or executable-procedure record. Its second action responded to the observed movement, but it did not imitate the demonstrated `set_experiment` workflow.

D made nine Actor calls before `UP`, then eighteen more before `LEFT`. It did record both experiments and used the measured 52-cell transition to make a precise second prediction, but it spent extensive calls rebuilding and revising helpers. It wrote no executable retained procedure. The evidence interface worked; it did not amortize discovery.

The final A trajectory demonstrates large stochastic variance: this baseline instance spent 23 Actor calls without acting, whereas the immediately preceding run's A instance completed two actions in ten Actor calls. That prevents precise treatment-effect estimates from single trajectories. It does not rescue C or D as efficient candidates: neither beat even C's own bounded objective, and D was plainly costly.

## The important implementation finding

The artifact called an “executable discovery demonstration” is fixture-generated and API-faithful, but its rendered prompt is not actually an executable transcript. It lists signatures, measured return values, prose such as “Decision 1,” and an action result. It does **not** show the model a concrete Python cell containing:

```python
grid = self.model_grid()
summary = self.grid_summary()
self.set_experiment(...)
self.submit_actions(...)
```

nor the corresponding second-decision cell. The renderer discarded the captured call arguments for most operations. Therefore this experiment tests a narrated trace of tool results, not the code-centric procedural demonstration the design intended. C's failure to adopt `set_experiment` is consistent with that gap.

## Decision

- Reject D in its current form; evidence plus the narrated demonstration compounds prompt and serial-call cost.
- Do not promote C; it completed but did not reduce work or adopt the full procedure.
- Retain the reasoning-off, context, summarizer-purpose, lifecycle, and telemetry mechanisms.
- Before another paid comparison, repair the demonstration renderer so the fixture emits literal, executable two-decision Python cells with captured outputs and a narrow update. Then test adaptation locally: changed appearance/control and a contrasting outcome must change the conclusion rather than invite copying.
