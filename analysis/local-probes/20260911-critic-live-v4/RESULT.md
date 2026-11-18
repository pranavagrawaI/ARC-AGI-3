# Live epistemic critic: role works, improvement not demonstrated

The critic produced a concise, valid review when given its own system role and the entire Actor request as quoted evidence. The Actor later acknowledged it and acted. This pilot did **not** demonstrate correction of an epistemic mistake, a better action, or faster decision closure. An injected self-review skill also did not establish those benefits.

## Observed contact

All arms used Qwen3.8-27B through the existing OpenRouter CoreWeave FP8 route, reasoning disabled, with the unchanged frozen submission-runtime-20260910-v6 files. The external harness used a declared Agent subclass for seed context and submission logging. Contingent continuation was disabled in every arm. Every replayed before/after grid matched the seven-step preserved LS20 prefix exactly. Each Actor started fresh with notes through historical Turn 7 and host measurements; neither the historical full conversation nor helper was restored.

| v4 arm | Actor calls | External critic calls | New actions | Time to first action | Stop |
|---|---:|---:|---|---:|---|
| Isolated critic | 8 | 1 | RIGHT | 94.28s | Actor call cap before second action |
| Injected skill | 8 | 0 | None | — | Actor call cap |
| Baseline | 2 | 0 | RIGHT | 22.32s | Next request exceeded conservative reservation cap |

Both RIGHT actions started from identical grids and produced identical successor grids: 52 changed cells, zero levels completed. The baseline is censored after two calls. These are independent sampled continuations, with different behavior even before review eligibility; the table is not an estimate of the critic's causal effect. In the earlier v2 baseline, eight calls produced no action, illustrating the variability directly.

v4 used 19 requests and cost **$0.1335364**. All live cards together used 30 requests and cost **$0.2101706**. Actual cost was below $1. Conservative reservations were not settled down to actual reported cost between calls, so they stopped the last baseline despite substantial unused actual-dollar headroom. This limits comparison; it is not a provider failure. The raw worker reports these host admission stops as BadGatewayError; `errors.jsonl` identifies the actual cap reasons.

## Invocation and context

The initial implementation intercepted the first `submit_actions` attempt before normal action publication. It could not help an Actor that never reached submission. v2 exhausted eight calls repeatedly inspecting the same state. In v3/v4 the host instead intervened before Actor request three, without requiring an action proposal. It retained the feedback as the final user message on subsequent Actor requests. Critic code never executed environment actions.

v3 retained the actual Actor message list and tool schema byte-for-byte at the JSON value level, appended a critic-role instruction, and set tool_choice=none. The critic failed its role/output contract: 384 tokens of prose and textual execute_python markup, truncated before valid JSON. It took 8.97s and reported **zero cached tokens**. No critique was delivered to the Actor; that card stopped. Equal message values do not prove identical provider tokenization or cache placement.

v4 gave the critic the v5 epistemic system instruction, no declared tools, and the **complete imminent Actor request as quoted JSON evidence**, plus the last executed Actor output. The audit confirms exact equality of that quoted request with the captured Actor request. This preserves information while separating instructional authority; it does not preserve the active Actor prefix.

The v4 critic returned valid JSON in **4.62s**, using **27,588 prompt tokens and 61 completion tokens**, costing **$0.0112182**, with **zero cached tokens**. Its assessment was no_concern: the two inspection steps did not establish a consequential reasoning defect. This is a plausible local judgment, not validation of the Actor's later models or all earlier notes. The response contains 35 words across its field values; an earlier progress update's 47-word count was incorrect.

Caching did operate on ordinary Actor requests in v2 (for example, 7,840 cached tokens on request two). Neither critic trial demonstrated Actor-to-critic cache reuse. OpenRouter exposes cached-token usage and describes provider/session routing in its [prompt-caching documentation](https://openrouter.ai/docs/guides/best-practices/prompt-caching). Cache affinity, role instruction placement and provider tool serialization require actual measurement; shared model weights alone establish none of them. These timings are OpenRouter/local-harness observations, not Kaggle SGLang latency measurements.

## What changed in observable reasoning

In v4 the Actor did not explicitly respond to the review on its first continuation. Six further Actor calls elapsed before the cell at response call 9 acknowledged: “Host review: no_concern — initial inspection is valid.” It then recorded movement and bar behavior as untested hypotheses, left the win condition unknown, and submitted RIGHT as a probe. That is acknowledgement plus action. Since the critic requested no correction, it does **not** establish repaired epistemics. Its movement prediction also differed from baseline, but the review did not prescribe that revision and the independent trajectories cannot attribute it to criticism.

The skill arm eventually wrote a claim/evidence/tested-scope/open-conditions table and kept the objective uncertain. It still committed no action in eight calls. Epistemic vocabulary and orderly notes are not evidence that reasoning improved. One sentence—“no alignment hypothesis is testable without knowing target”—also needs scrutiny: uncertainty about a target need not rule out constructing and testing candidate targets. This is a candidate reasoning issue for a future scoped review, not a factual correction established by this pilot.

The baseline produced RIGHT on its second call without review. Its stated prediction was a one-column shift; the critic arm predicted five columns. Same action is not same belief. Neither the movement difference nor the action counts prove treatment efficacy.

## Contract to retain and test next

1. **Independent role, shared evidence.** Use the isolated critic system instruction now. Treat Actor history, including its prompts, as evidence. Keep the critic without an action dispatcher. Optimize a stable critic prefix across reviews only after role reliability; the active shared-prefix shortcut failed this screen.
2. **Host invocation at a consequential commitment or failed closure.** Review before long batches or abandonment of an option, and when repeated inspection has no stated decision-relevant uncertainty. A model-invoked skill can supplement this but cannot be the only route out of fixation. The two-call trigger here proved reachability, not optimal timing; it arrived before a demonstrated inferential error.
3. **One exact finding, permission to find none.** Quote the inference, identify how evidence has been overextended, and state the smallest revision to belief or commitment. Do not recount pixels, solve the game, demand complete certainty, or invent an objection to a bounded probe. Scope no_concern to the reviewed artifact.
4. **An explicit, bounded Actor response.** Require incorporation or disagreement with reasons and the resulting belief/decision, rather than more general inspection. Keep the outcome connected to a review ID and expire the review when the claim or evidence changes. Acknowledgement alone earns no correctness credit; evaluate the ensuing reasoning and action. Do not require changing an already defensible action.
5. **Next efficacy test: a real faulty commitment.** Fork the same captured decision context into reviewed and unreviewed continuations, with matched action/call budgets and an independently specified inference-error rubric. Measure correction, unsupported criticism, time to action and actual successor. This pilot's no_concern example cannot answer whether criticism repairs an epistemic mistake.

A short critic response does not imply a short control loop: here its 4.62s was followed by many Actor calls. The skill is cheaper only in the narrow sense of avoiding a separate review call; it did not achieve decision closure in this trial. Neither is ready for frequent deployment on this evidence.

## Reproduction and integrity

Run `deployment/kaggle/.venv/bin/python experiments/probes/epistemic-critic-local/audit_live_pilot.py analysis/local-probes/20260911-critic-live-v4` to recompute mechanical evidence without inference. `ACTOR-CELLS.json` extracts executed-cell proposals from preserved model responses; raw host actions establish which proposals actually reached the environment. Successful submit_actions raises NOOA's return-result signal, so its success does not produce the wrapper's ordinary-return event.

Six focused tests passed, including declared-hook compatibility with the real NOOA class, commitment suppression in submission mode, ordinary rejection preservation, shared-prefix equality, and complete quoted-context isolation without tools. The repaired wrapper also reached ipc/agent_ready against a non-network endpoint before paid testing. All pilot processes retired. Frozen runtime files and unrelated worktree changes were left intact.

Prior cards are immutable: v1 failed before any inference on prohibited dynamic method attachment; v2 exposed the unreachable submission trigger; v3 exposed role leakage. Their raw outputs and v2/v3 source snapshots remain in sibling directories. This is an experimental local harness, not an integrated production critic. No Kaggle run or deployment occurred.
