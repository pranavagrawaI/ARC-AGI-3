# Current solver diagnosis

## Judgment

The current solver can run a real ARC-AGI-3 episode, form and revise an
executable local mechanics model, and batch actions. It has not yet shown
level-solving ability. The best sustained public run made 100 actions on
`ls20`, changed the grid 99 times, and achieved zero completed levels. Its limiting
factor is not hidden reasoning tokens, Python execution, IPC, or the game
environment. It is serial Actor inference: the model repeatedly asks to inspect
or reinterpret its own work before committing an action batch.

The ratings below use the six frontiers recorded in board thread 119. They rate
demonstrated current capability on a 1--10 scale, where 10 means a reliable
competitive advantage on that frontier. A high score for a support frontier
does not imply gameplay success.

| Frontier | Score | Judgment |
| --- | ---: | --- |
| 1. Speed and Action Economy | **3** | The worker batches actions cheaply, but the Actor spends too many serial calls earning a batch: 0.65 executed actions per Actor call in the long run. |
| 2. Tools, Math, and Representation | **5** | The model has real NumPy, grid, trajectory, helper, and action tools, and can make partial executable models. It repeatedly misidentified salient objects and never represented a goal or terminal condition. |
| 3. Harness and Interface | **7** | The official host path ran 100 actions with exact request accounting, per-action transitions, and clean containment. A stale post-run chassis pin still made the probe invalid, and animation/context-budget semantics remain imperfectly exposed. |
| 4. Metacognition and Judgment | **2** | The Actor notices prediction mismatches and sometimes repairs its helper, but it did not identify the level objective after 100 actions and continued costly local-mechanics loops without a goal-discriminating experiment. |
| 5. Learning, Evaluation, and Retention | **4** | Source hashes, raw transitions, prediction records, and local replay exist. There is no automatic candidate comparator, held-out episode suite, or evidence that a retained mechanism improves levels. |
| 6. Portfolio and Resource Allocation | **4** | Runs have cards, bounds, accounting, and model-gated investigations. Spending 50 minutes and 153 Actor calls for zero levels on one game shows that the current behavioral loop is still a poor consumer of scarce inference time. |

These are deliberately asymmetric. The harness is substantially more mature
than the judgment loop it carries.

## What ran

The strongest direct evidence is the public `ls20` run from
`q6-summary-boundary-20260906` on an NVIDIA RTX PRO 6000. It used Qwen 3.8 27B
FP8 through SGLang. Thinking was explicitly disabled and every recorded request
reported zero reasoning tokens. The game was `ls20`, not a cross-game benchmark.

| Measure | Observed value |
| --- | ---: |
| Executed actions | 100 |
| Submitted action batches | 40 |
| Submitted actions | 101; the cap prevented the final queued action from executing |
| Grid-hash changes | 99 |
| Levels completed | 0 |
| Final host state | `NOT_FINISHED` |
| Actor model calls | 153 |
| Summarizer model calls | 2 |
| Actor model wall time | 2,979.78 s |
| Summarizer model wall time | 111.14 s |
| Gameplay wall time | 3,005.16 s |
| Peak recorded prompt | 107,615 tokens |

The official launcher returned zero. The outer probe assigned exit code 74 only
after gameplay because its session validator required an obsolete chassis name.
The isolated manifest-based validator repair accepts the saved run. This means
the run is valid evidence about gameplay and latency, but it is not a qualified
deployment result.

The retained earlier runs point in the same direction:

| Run | Actions | Actor batches / decisions | Model time | Levels | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| q6 speed probe | 10 executed; 16 submitted | 6 | 1,325.0 s | 0 | 99.5% of gameplay wall time was model time; one decision made 56 calls and took 720 s. |
| Prediction-feedback long run | 19 | 12 | 1,256.3 s | 0 | 60 Actor calls; the feedback mechanism was observed but not shown to improve solving. |
| Summary-boundary long run | 100 | 40 | 2,979.78 s Actor + 111.14 s summary | 0 | Best sustained behavioral evidence; summary boundary worked twice, but no level progressed. |

## The rate-determining step

The host path is:

```text
MyAgent -> LocalSolver -> nooa_worker -> headless -> MdArcSolverAgent -> SGLang
```

Once the Actor submits a batch, `nooa_worker` returns queued actions on later
host decisions without another model request. It only wakes the Actor when the
queue is empty. The cheap path therefore exists; the solver seldom earns a
long enough queue.

In the 100-action run, 40 action batches produced 100 executed actions:

- 26 single-action batches;
- 9 five-action batches;
- one three-action batch, two four-action batches, one nine-action batch, and
  one ten-action batch.

That is 2.5 executed actions per Actor reasoning turn, but only 0.65 executed
actions per individual Actor model call. The distinction matters: a reasoning
turn is itself an uncontrolled sequence of model -> Python -> model crossings.

The 100-action call trace divides as follows:

| Generated Python-cell purpose | Actor calls |
| --- | ---: |
| Submit an action or batch | 44 |
| Inspect current state or prediction feedback | 28 |
| Re-read or diff the trajectory | 27 |
| Rewrite the world-model helper | 16 |
| Reload or test the helper | 14 |
| Write hypotheses or a step log | 11 |
| Other grid analysis | 11 |
| Read helper source | 2 |

The slowest observed reasoning turn in this trace was turn 30: 23 Actor calls
and 506.99 seconds before a single submitted `RIGHT`. It repeatedly inspected
collision geometry, rewrote the helper, ran local path searches, reconsidered a
12-step strip-consumption plan, wrote a note, and finally had to submit one
action because the new helper version had no successful prospective check.
Turn 19 used 16 calls and 297.67 seconds for another model-revision loop.

This is model-mediated micro-stepping, not expensive Python. Individual actor
requests were commonly in the 5--25 second range. Calling a larger/faster
serving stack can improve a request, but cannot remove the multiplicative cost
of 16--23 requests before one action.

## What the Actor learned, and what it did not

The Actor did do genuine measurement and model building. It found local
regularities for a color-12/color-9 block and a color-11 strip: motion in
five-cell increments, blocking at some boundaries, and a strip disappearance /
wrap event. It made 16 distinct committed helper source versions in the
100-action trace.

It did not discover the objective of `ls20`. By step 87, after its second strip
wrap observation, its own retained record still asked for the winning condition.
Its final helper encoded only the block position and strip extent; it had no
objective, terminal predicate, or reliable movement-validity representation.
The final ten actions were repeated `RIGHT`s to advance the strip. They did not
produce a level transition.

The correct semantic reading is therefore: partial mechanics knowledge,
unestablished objective knowledge, and no evidence that the agent was near a
level completion. Ninety-nine changing frames are not progress.

Prediction feedback is useful but has a narrow meaning. Across observed
commitments in the 100-action trace, 23 were matched and 16 mismatched under
the Actor's frozen encoder. That is self-consistency under its chosen latent
state, not proof that its representation preserved a decision-relevant object
or goal. In the trace, a mismatch could also occur during a batch while later
queued actions still executed. The current worker stops queues only for terminal
state, level change, or exact no-impact.

## Support mechanisms: what is sound and what remains unresolved

The following are demonstrated by the live run or its saved-data replay:

- Public observations, per-action reconciliation, transitions, model-call
  accounting, and process containment worked for the 100-action episode.
- The two live summary calls each verified their exact source history, completed,
  and gameplay continued afterwards. Together they were about 3.7% of gameplay
  wall time; removing summarization is not the primary speed improvement.
- Actor and summarizer call accounting reconciled: 155 solver calls plus one
  warmup matched 156 server accesses.
- Reasoning was disabled in request configuration and zero reasoning tokens
  appeared in all 155 recorded calls.
- The earlier unawaited state-queue getter could make generated code accumulate
  coroutine objects until it reached roughly 21 GiB RSS. The current path uses
  synchronous `latest_state()` / `trajectory()` access and applies worker memory
  and computation bounds.

The following should not be promoted to stronger claims:

- `ARC_LLM_CONTEXT_WINDOW=32768`, `ARC_LLM_MAX_TOKENS=4096`, and zero retries
  were written into notebook environment setup, but current LLM construction
  does not demonstrably apply all of those values. The observed 107,615-token
  prompt proves that the named 32k value was not a full-request bound.
- Two successful summary calls establish serialization and liveness, not that
  the summaries retained the distinctions required for later solving.
- The 100-action episode is one public game. It is evidence about this solver's
  behavior on `ls20`, not a general performance estimate.
- The prediction-feedback repair submitted legal actions in local replay, but
  no matched live control established an effect on levels, speed, or score.
- MTP is a separate serving experiment. It must not be credited with a solver
  behavior improvement until a matched game trial separates it from the
  decision-loop changes.

## Smallest path toward a fast, useful solver

The next improvement should reduce unnecessary model crossings while preserving
the ability to learn an unknown objective.

1. Build a deterministic current-state packet from records already maintained:
   latest grid diff, last prediction result, known invalid first actions, helper
   version, and the explicit unresolved experiment. Keep raw grids and the full
   trajectory available on demand.
2. Make routine turns one composite cell: inspect the packet, test or make the
   smallest helper change if necessary, and submit a discriminating action.
   Measure calls and model wall time separately for routine, repair, and
   objective-discovery turns.
3. Add per-action interruption for a committed prediction mismatch. A correct
   batch remains free of further inference; an incorrect prefix returns control
   before more actions spend the environment budget.
4. Require an objective conjecture or a named objective-discriminating probe
   once mechanics loops cease changing the hypothesis. Do not require a complete
   game reconstruction before exploration.
5. Compare the new loop against the current loop on held-out public transitions
   and then matched game episodes with fixed model, serving configuration, action
   budget, and wall-time budget. Keep the change only if it improves an outcome
   that matters: level progress, useful prediction coverage, or action/model
   cost at equal progress.

This order follows the evidence. It treats faster serving as an accelerator for
a good decision loop, not a cure for an aimless one.

## Evidence

- Board thread 119: authoritative definitions and ordering of the six frontiers.
- `analysis/deployment-probes/20260906-q6-summary-boundary/RESULT-run.md`:
  live run, request accounting, summary behavior, and host outcome.
- `analysis/deployment-probes/20260906-q6-summary-boundary/exit-review/RESULT.md`:
  post-run exit-74 cause and saved-data validator repair.
- `analysis/deployment-probes/20260906-q6-summary-boundary/raw/nooa-join-public3/solver-workspaces/ls20-8md_11v1/ipc/model_calls.jsonl`:
  call timing and generated Python cells.
- `analysis/deployment-probes/20260906-q6-summary-boundary/raw/nooa-join-public3/solver-workspaces/ls20-8md_11v1/ipc/actions.jsonl`:
  submitted batches and predicted paths.
- `analysis/deployment-probes/20260905-q6-speed-probe/RESULT.md`:
  earlier latency distribution and 56-call decision outlier.
- `analysis/deployment-probes/20260906-q6-prediction-feedback/RESULT.md` and
  `long-run/RESULT-LONG.md`: bounded repair replay and 19-action behavioral
  observation.
- `analysis/competitor-transfer-20260906/CURRENT-STATE-UPDATE.md`:
  source-checked current architecture and limitations.
