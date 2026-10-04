# q6 solver and Builder examination

The useful lead is **a failure to turn investigation into an executable candidate before the deadline**, rather than pervasive tool execution failure. The Builder was given a 240-second allocation, but neither its time/call budget nor its existing finishing mode was exposed to the model. Both passes were still inspecting when cancelled. This is a source- and trace-supported explanation for the observed failure path, not proof that one prompt change will improve score.

The source is the frozen q6 runtime; artifacts are hash-checked against its evidence index by `experiment/analyze_behavior.py`. Reproduction: `python3 solvers/tycho_dfranzen/experiment/analyze_behavior.py solvers/tycho_dfranzen/experiment/gpu-q6`. Detailed counts, role usage, writes, request samples and evidence hashes are in [BEHAVIOR.json](BEHAVIOR.json).

## What the roles actually did

| Game | Completed calls: Actor / Builder / Scribe | Observed solving behavior |
| --- | --- | --- |
| AR25 | 151 / 0 / 21 | Actor learned mirror/coverage rules, wrote search and render/simulation helpers, and completed seven levels. Its level-six decision at sequence 178 contains 26 Actor calls and successive helper revisions before acting. |
| LS20 | 105 / 40 / 6 | Actor solved the first level, delegated full model construction twice during the second, then continued solving itself. Two levels completed; neither Builder invocation yielded an accepted model. |
| SB26 | 92 / 0 / 3 | Actor inspected the color-box puzzle and its animations, wrote `gz.py` and a beliefs note, and kept testing ordering hypotheses. One level completed. Decisions 35 and 48 required 23 and 21 Actor calls respectively. |

These runs show executable analysis inside the Actor as well as the optional Builder path. Counting Builder invocations alone would miss the Actor's useful computation. None of the captured tools called `dfr.segment` by name. AR25 did use NumPy/SciPy in model-written helpers. Infrastructure compatibility and useful computation therefore need separate descriptions; the preloaded donor segmentation helper's causal benefit is unestablished.

## Builder passes

| | First pass | Second pass |
| --- | ---: | ---: |
| Decision sequence / turn in second level | 30 / 0 | 72 / 42 |
| Completed model calls | 22 | 18 |
| Completion tokens | 21,809 | 24,352 |
| Prompt tokens, first -> last call | 6,918 -> 62,856 | 6,766 -> 48,678 |
| Tool calls | 23 | 18 |
| Successful model/helper writes | 0 | `features.py` only |
| Result | Deadline; no candidate retained | Deadline; helper rolled back |

Across both passes, 38,014 of 46,161 completion tokens are reasoning tokens. Most tools succeeded: the only rejected tool invocation was `write_file(path=features.py, code=...)` instead of `content=...`; the next call corrected it. No `world_model.py` edit occurred. No changed model was run through the verifier, and no fresh model notes or valid report reached the Actor.

The first pass read notes and the template, inspected frame/transition keys, dumped changes, inspected player coordinates, socket/badge pixels, budget changes, terminal pixels, animation metadata, terrain and rings. It spent substantial reasoning on unobserved budget-exhaustion behavior and exact rendering concerns. Those questions can matter; the failure is that inspection never produced an incremental executable model to test. Its final completed tool read the beginning of `verify.py` rather than executing verification of an edit.

The second pass started afresh, reread the notes/template and reconstructed features again. It wrote a useful feature module, then continued frame dumps, transition tables, terminal inspection and hypothetical dynamics. The first pass's investigations were preserved in external traces, but no valid report or canonical workspace update survived. The second pass therefore had no accepted model or Builder-authored semantic checkpoint to build on. The Actor's broad “verified mechanics” assignments also contained open or changing hypotheses; these are inputs to test, not facts the Builder should blindly encode.

## Why the deadline won

1. **Budget invisible to the model.** Frozen `src/hybrid/agent.py` constructs `BuilderConfig(max_tokens=12*1024, temperature=.7)`. `show_budget_telemetry` defaults to `False`. `BuilderStrategy._prepare_builder_round` immediately returns the full tool set in that case. Captured calls 53 and 116 contain zero `BUILDER BUDGET` notices; all six workspace tools remain offered. The host's hard 240-second cap still runs.
2. **Finishing happens after construction.** The loop allows up to 40 model rounds. The notes checkpoint and final-report phases run only after that loop ends or the model stops using tools. The hard deadline cancelled these passes after 22 and 18 completed responses, before either finishing phase. A “guaranteed fresh report” on ordinary return is not a guarantee under cancellation.
3. **Whole-pass rollback makes unfinished work unavailable.** Cancellation restores the workspace. Noncanonical retention applies to changed `world_model.py` bodies. There was no changed body in either pass, so retention could not help; the second pass's feature module was rolled back. This is intentional authority protection with an operational cost, rather than an accidental artifact deletion.
4. **Inspection expands context without a deliverable.** The first pass has a ~16k-character tool result; the second starts by printing ~11.8k characters of notes/template. The same categories are repeatedly inspected. Builder cached tokens were 1,145,088 out of 1,216,644 total reported prompt tokens (~94.1%); these traces do not establish uncached prefill as the main bottleneck.

Simply turning on budget telemetry is insufficient as a guaranteed fix. Its commit mode offers only writing tools and requests `tool_choice=required`, but the construction loop still has no transition to finishing on entering that mode or successfully checkpointing. It can keep demanding edits until its step limit/deadline. The “finishing calls reserved” notice does not itself reserve wall time. That branch was disabled in q6, so it did not cause q6's failure; it limits the obvious proposed remedy.

## Independent checks and limits

The Actor's last LS20 reasoning predicted that DOWN from (45,54) into the socket would win. Host action 157 instead left the player, budget bar, level count and public state unchanged. That is a directly refuted prospective prediction. It does not establish the complete rule responsible, or a Builder recovery interval. Inspection of pixels alone is insufficient to assign a unique cause.

AR25's notes describe planned solutions and its helper code provides executable computation. Those observations support examining modelling/search as a lever, but neither notes nor seven completions establish a causal advantage over a matched donor run. SB26's inability to settle its ordering hypothesis likewise cannot be attributed to the absence of a Builder without a contrast.

First-shot Builder mechanism correctness and Builder recoverability remain unmeasured because neither pass returned an accepted executable model or checked prospective prediction. Error-free tool execution, argument repair and retrospective agreement are not substitutes for those measures. There is no matched score-improvement verdict.

## Recommended next task

Build and locally test **a bounded construction/finish lifecycle at the same 240-second cap**: expose the authoritative remaining budget; externalize a small falsifiable candidate early; stop investigative rounds while sufficient finishing time remains; run the verifier and persist a report/checkpoint; preserve unfinished bodies only as noncanonical evidence. The Actor should retain its choice to invoke construction.

Use a loopback model fixture that keeps investigating to demonstrate the lifecycle transition and cancellation/rollback behavior before spending GPU time. The subsequent mechanism comparison should distinguish finishing control from the modelling algorithm and role delegation. Keep official score at equal resources decisive; first-shot predictions and recovery are supporting measurements.

The accounting bug is repaired and locally qualified separately in [the repair review](../../evidence/deadline-accounting/REVIEW.md). No Builder reasoning policy was changed during this examination.
