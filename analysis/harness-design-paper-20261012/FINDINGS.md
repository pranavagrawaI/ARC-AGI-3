# arXiv 2609.20804 applied to the B1 harness (2026-10-12)

Paper: Fan et al., "An Empirical Study of Harness Design for Coding Agents" (17 Sep 2026). A fixed ReAct loop is used while three components vary:
context management (T0 none, T1 elision, T2 elision+recall, T3 summarization, T4 elide-then-summarize), planning (a persistent `update_plan` todo
re-injected each turn outside history), and action space (predefined tools vs bash-only). The models are Nemotron-3 30B/120B/550B and Mistral-3.5-128B,
at 32K–128K windows, on SWE-Bench Verified and Terminal-Bench 2.1.

## Paper findings, and whether each transfers to us

| Paper finding | Mechanism the paper identifies | Our regime | Transfers? |
|---|---|---|---|
| Context management (CM) helps when the window binds: managed–T0 gap 35.7 → 2.7 pp from 32K to 128K | The benefit is preventing overflow termination. Trajectories get longer, but behavior does not change. | 128K window with hysteresis drains. B1 pair: **57 drains over 7,973 requests (0.7%), 1.1 per game**, 33/50 games drain at least once. A drain loses context; it does not end the episode. | **No.** We are in the paper's 128K row, where CM tiers are indistinguishable. The drain cost was measured earlier at +19% tokens for the 3 requests after a drain, about 0.1% of run tokens. |
| T4 (elide before summarizing) is cheapest at equal accuracy | Cheap elision avoids LLM summary calls. The cost saved is input tokens. | Our cost is decode time: GPU step-time-bound, prompt tokens 94% cached. Bulky tool output is only ~10% of late context; 51% is past thinking. | **No.** The saving the paper measures is not our binding resource. Rolling summaries and `STUB_DEAD_REASONING` already exist and are off. |
| Recall (`recall_event`) is almost never used: median zero calls | Models do not retrieve what is not in view. | `solvers/m2_memory` `save_data`/`load_data` is the same opt-in retrieval pattern. B1's level-opening pin is the opposite: automatic retention, the ARC analog of the paper's verbatim preamble. | **Yes, as a prediction.** Expect low `load_data` uptake. If m2_memory is ever run, gate on its uptake counter before reading levels. The pin design is consistent with the paper. |
| Planning adds accuracy for weak models (prevents early stopping) and only saves cost for strong models (cuts post-edit verification), with −0.4 to −2 pp accuracy | It changes where trajectories stop. | Our episodes cannot stop early, and ARC has no post-success verification phase, because a solved level advances by itself. Our harness's equivalent is `ARC3_MEMORY_SECTIONS=slim` (world model + plan); dfranzen got his best result with it **off**. Prior model-written memory nulls: Duck note audit, KF arm R. | **No.** Neither of the paper's mechanisms has a target in our loop. |
| Bash-only wins for capable models (denser composite calls, fewer re-patches); predefined tools help weak ones | It changes action granularity. | We already use a single composable Python tool: several `action()` calls per cell, plus helpers in the same namespace. Tool errors are 8%, mostly fixed in m2_evidence/B1. | **Already adopted.** It argues against adding separate predefined tools. |
| Stuck detection (5 identical calls → reminder) | Fixed substrate in the paper; not ablated. | Repeats of known no-ops are 1% of actions (STUPIDITY-MAP §7.4). | **No.** There is no target. |

## Decision

The paper changes nothing in B1. Its results describe the corner of the design space we already sit in: a large window with cheap rule-based
trimming, no planning scaffold, no recall machinery, and one composable code interface. Its effects come from preventing termination and saving input
tokens, and neither binds us. Our binding deficit is rule induction and closure within a level (oracle-hint, stuck-levels notes), which the paper does
not study. No arm from it clears the ≥20% levels/h bar.

Reusable takeaways:
1. **Before adding a harness component, measure the failure it prevents** (overflow rate, early-stop rate, verification share) in our own logs. If that
   failure is rare, predict null. This would have pre-screened several past arms.
2. **Opt-in retrieval goes unused.** Prefer automatic injection (the pin) to model-called memory APIs (m2_memory).

Drain counts: `solvers/m2_b1/experiment/pair-1/b1-r{1,2}/output/artifacts/*_tool_runtime_state_b1.json` (the `trims` field).
