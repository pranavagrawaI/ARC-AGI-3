# Real critic endurance: unsupported approval despite visible counterevidence

Two fresh LS20 games ran through the real adapter, worker, NOOA Actor, action publication and environment loop. Across **12m11s**, the arms executed **19 actions** and completed **zero levels**. The run produced an unambiguous failure in a critic's justification: it endorsed a reachability conclusion contradicted by evidence in its own input. The Actor subsequently repaired the search independently.

| Arm | Duration | Actor calls | Critic calls | Executed actions | Levels completed | Actual cost |
|---|---:|---:|---:|---:|---:|---:|
| Critic | 5m48s | 40 | 9 | 9 | 0 | $0.4049836 |
| Baseline | 6m23s | 44 | 0 | 10 | 0 | $0.3913790 |

Total: **93 requests, 93 responses, $0.7963626**. Both arms stopped at their predeclared $0.50 spend-admission limits: the conservative reservation for the next large request could not fit. They did not reach the ten-minute or64-action ceilings. Completed requests released unused reservations; both final unsettled balances were zero. The worker's RateLimitError at shutdown represents the host's budget refusal, not provider rate limiting or an unexplained crash.

The arms started from identical fresh grids with separate workspaces and no injected historical notes. They used the same frozen v6 runtime and generation settings; only the critic arm received reviews. This is one sequential game per arm with different sampled trajectories and extra inference in treatment. Neither the action counts nor the completion counts establish a causal ranking or separate-role advantage. First action occurred at38.64s with review and95.23s without it; this variability is likewise not a causal speedup estimate.

## The decisive observed case: review8

The evidence and its temporal boundaries are preserved in `CASE-R8.json`.

1. Actor call38 labeled its code a full BFS. It added discovered positions to `visited` but never added them to the queue. Only the initial state was expanded, producing four positions.
2. Call39 inferred that the block could not move UP from(40,34), then checked individual placements. Its actual tool output said `From (40, 34): UP -> (35,34): OK`. That line was visible in the critic's captured input. Some unrelated middle output was truncated, but this counterexample was retained.
3. Review8 returned no_concern and stated: **“The BFS finding only 4 reachable positions is a valid consequence of the current wall layout.”** It approved the conclusion despite an explicit edge leading beyond that set.
4. The Actor repeated the broken loop in call41. In call42 it noticed that an allowed deeper position had not been expanded, added the missing queue insertion, and obtained **37 reachable positions**. The captured execution output confirms the result.
5. Later real UP actions moved the color12 region to rows35–36,30–31 and25–26, beyond the initial one-expansion result. These later observations corroborate the repair; they were not supplied to the earlier critic.

The distinction matters: the critic did not need to debug BFS or solve the game to notice that the reachability conclusion and the available placement check disagreed. Its stated justification failed an epistemic consistency test. The no_concern verdict alone could acknowledge that the Actor was actively diagnosing a problem; the rationale certifying the four-position result cannot be defended on that basis.

The Actor's repair was real and informed subsequent actions. **It was not supplied by the critic.** The Actor used the word incorporate while explaining a correction that contradicted the critic's approving rationale. Agreement labels would have scored this interaction misleadingly.

## Review delivery, uptake and retention

Ten review opportunities arose; the tenth was refused before inference at the spending gate. Of nine actual critic responses:

- **Six valid responses were delivered**, all no_concern.
- **Two proposed concerns exceeded the word budget** and were not delivered.
- **One no_concern used an ellipsis-joined, non-exact quotation**, failed source binding and was not delivered.

All six valid reviews entered the immediately following Actor request. No invalid review entered the Actor context. Five of six immediate Actor responses explicitly used incorporate; review8's first continuation did not. Acknowledgement is demonstrated more often than decision closure, and no delivered concern requested a correction whose retention could be tested.

The invalid concerns also matter substantively. Review3 questioned causal attribution from one transition, which can be reasonable, but invoked non-intuitive directional mapping as a defect and prescribed UP/RIGHT instead of LEFT without establishing why that replacement was necessary. Review6 said **“DOWN has never been tested”**, although DOWN was the first action. There was a legitimate narrower issue: the initial DOWN produced no movement of the color12 region, so a DOWN test from a new position could still be informative. A boundary-limited test is not absence of a test. Neither response should be treated as wholly grounded criticism simply because a proposed probe might be useful.

Reviews4 and9 used language such as accurate tracking and actual measured reachable positions. These formulations risk promoting Actor interpretations or computations into verified game knowledge. Review5 more usefully distinguished helper preflight from demonstrated dynamics. Detailed grades, including these qualifications, are in `REVIEW-GRADES.json`.

Several reviews had four or more subsequent real decisions, so this was longitudinal contact. However, **retention of a critic-requested correction was not demonstrated**, because no concern was successfully delivered. Independent Actor self-repair did persist into notes and later actions. Unsupported approval is observed; causal harmful persuasion is not established by this unpaired run.

## Speed and context reuse

The nine critic requests took **39.78s total**, averaging **4.42s**. Their prompt sizes grew from20,832 to67,291 tokens; generated completions ranged from67 to181 tokens. Later critic requests reported cached prompt tokens:

| Review | Prompt tokens | Cached prompt tokens | Request time |
|---|---:|---:|---:|
| 1 | 20,832 | 0 | 4.91s |
| 2 | 26,819 | 15,680 | 3.43s |
| 8 | 62,450 | 54,096 | 3.82s |
| 9 | 67,291 | 54,096 | 3.95s |

An independently instructed critic can therefore retain useful cache reuse across repeated calls on this endpoint. This does not isolate Actor-to-critic sharing or quantify the latency benefit of caching: there was no cache-disabled comparison. Timing includes OpenRouter/network overhead and does not establish Kaggle SGLang performance.

The full current Actor request was supplied as quoted critic evidence, with a separate system role and no critic tools. Audit checks confirm exact context equality for all nine reviews. The context includes the Actor's existing truncation of some tool outputs; complete Actor context is not a claim of an unabridged environmental evidence record.

## What this run licenses

The run establishes real repeated invocation, valid feedback delivery, several later decisions, cache reuse across critic calls, and a concrete failure to respect contradictory evidence. It does not establish improved decisions, retained critic-driven repair, level-solving benefit, or that a separate critic role is superior to extra Actor inference.

The next warranted focus is the critic's distinction between **an observed program output and a justified conclusion about the game**, together with explicit contradictory evidence. The captured review8 boundary supplies a substantially sharper diagnostic than the earlier ambiguous alignment sentence. No new architectural layer is justified by this result, and no further inference was run under this card.

## Integrity and reproduction

Seven focused preflight tests passed, including reservation settlement and unknown-cost containment, repeated-review admission, source binding and semantic/execution-status separation. The frozen runtime hash remained `061369c045635581ac5e4d9fea99d532722f3a3e5d0c4ce93b4b75475a7876ea`. Frozen experiment-source hashes matched after execution. Neither arm had an unresolved transport or accounting failure. Generated actions passed through the original submission gates; critic output never executed tools or environment actions.

The runner recorded full requests, responses, review contexts, host before/after grids, and helper/notes snapshots before each Actor request and at cleanup. No historical evidence was seeded, no arm received the other's outputs, and no Kaggle launch or deployed-file change occurred.

Recompute mechanical evidence with:

```bash
deployment/kaggle/.venv/bin/python experiments/probes/epistemic-critic-local/audit_endurance.py analysis/local-probes/20260911-critic-endurance-v1
```

`AUDIT.json` reconciles counts, costs, context equality, valid/invalid delivery and hashes. `REVIEW-TIMELINE.json` links each review to its next emitted Actor response and subsequent host-decision opportunities. `ACTOR-CELLS.json` preserves emitted code; successful execution is checked against tool outputs and host actions, not inferred from source text alone. `CASE-R8.json` separates evidence available before review from later repair and action corroboration.
