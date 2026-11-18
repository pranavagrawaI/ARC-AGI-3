# OR-CONSTRUCTOR-REASONING-PAIR — 20260916T185550Z

Status: PREPARATION; no spend authority. Independent exact-byte PASS, Hermes release, and explicit credential-use approval remain required. Sentinels: OR-PAIR-REVIEW-BEFORE-SPEND. PM operational rulings in `rulings.md` govern this card.

## Question / posture / contrast

Question: On the same already-executed CD82 prefix, does explicitly enabled hosted reasoning change whether the bounded constructor reaches a valid source/visible-ABI freeze or repairs a failed submission?
Posture: bounded descriptive protocol pilot, one episode per condition; not a gameplay or dynamics evaluation.
Contrast: independent off episode FIRST, on episode SECOND, both `qwen/qwen3.8-27b` through OpenRouter endpoint `reka/fp8`, FP8, no fallback. Off `{enabled:false,exclude:false}`; on `{enabled:true,effort:"medium",exclude:false}`. No warmup or paid preflight. No crossover/repeat or endpoint replacement.

## Prediction

The on arm will provide nonzero provider-reported reasoning tokens with returned reasoning evidence; off will report exactly zero and no reasoning payload. Directional, weak construction prediction: on will reach an accepted ABI freeze at least as often as off, or repair a failed source submission within six requests. With one ordered pair, either outcome is descriptive, not an estimated treatment effect; extra reasoning may instead exhaust the requested output/time budget.

## Shared evidence and prompt

`prefix.jsonl` is ONLY the first executed R10a B transition: step 1 RIGHT. `PREFIX-BINDING.json` binds the original transitions file and original first constructor wire request. Original compact visible packet equality is independently checked in preparation. No later step, prior candidate, previous failure, diagnostic explanation, or solution is placed in either arm's prompt.

Both arms use identical adapted system text, user evidence, original tool schema/source representations, constructor control loop, inspection, generic sealed ABI evaluator and actual feedback. Original list/text wording and stringified-list behavior remain unchanged. Shared transport-only deltas are in `engine.diff`: replace stale reasoning-fatal/local-GPU disclosure; replace Kaggle relay/game envelope with hosted envelope; preserve reasoning continuation; replace first request's rounded wall telemetry with exactly 240 for start parity (actual admission clock is still measured). The inherited conservative dollar proxy remains identical and explicitly separate from actual billing. This is same-adapted-prompt comparison, not byte-identical R10a prompt replay.

Protocol projection retains assistant null content, opaque call IDs/arguments, ordered reasoning_details (including signatures/index fields) or, when details absent, the returned reasoning string. It drops response-only tool index/refusal from outbound messages. No reasoning is injected into tool arguments/user prose. Both arms use the same transformation. Subsequent evidence/feedback may diverge based on actual generated tool calls.

## Measure / raw evidence

- `live/<arm>/child/raw/NNN.request.json`, `.response.raw`, `.receipt.json`: exact outbound body, raw response, hashes, HTTP status, latency, requested cap, reservation, usage/cost, finish reason and reasoning evidence. No authorization headers or credentials retained.
- `live/ledger.json`: every possibly-dispatched attempt, known actual dollars vs unresolved exposure; separate arm token/call sums. Reasoning is a component of completion tokens, never counted twice.
- `live/<arm>/child/PROGRESS.json` and `tool-events.jsonl`: tool order, recognized inspections/submissions, actual execution feedback, failed submission and repair; complete constructor transcript persists incrementally.
- Root `live/<arm>/RESULT.json` / `FROZEN.json`: timely promoted freeze only. Child files after failed/deadline exit are forensic, not authoritative. Freeze establishes complete-source execution and visible ABI replay only; commitments remain unscored.
- `live/<arm>/SUPERVISION.json`: local elapsed time, exit code, hard cutoff and reap. Local kill does not prove provider cancellation.
- Provider-reported off requires valid explicit reasoning count 0 and no payload on every call. On requires at least one nonzero count with returned compatible reasoning evidence; absent/mismatched counts/payload stop invalid. Accepted API flags alone are insufficient. Routing evidence must report Reka, matching model, direct strategy, attempt 1, one selected candidate, no pipeline transformations. Exact endpoint identity is the strict singleton outgoing tag plus compatible returned routing evidence, not a claimed physical worker/checkpoint hash.

## Numeric cap

Pair: maximum $2 USD admission budget. Each arm: maximum 6 paid attempts, 24576 reported completion tokens including reasoning, 600000 reported prompt tokens cumulative, 10 recognized inspections, 4 recognized submissions, 240 seconds soft / 245 seconds hard local supervision. Every request asks for at most 4096 combined completion tokens including reasoning (engine may reserve less in commit phase); observed overrun is invalid and stops pair. No guaranteed provider hard ceiling is claimed.

Before EACH dispatch reserve serialized complete request UTF-8 bytes + 4096 framing tokens at $0.214/M input, plus endpoint maximum advertised completion allowance 131072 tokens at $2.55/M output. Completion exposure alone = $0.3342336. This prompt bound is a conservative engineering reservation, not a certified hosted tokenizer mapping; usage exceeding it stops invalid. Require reserved prompt + requested output <= endpoint context 262144, cumulative prompt/completion admission limits, and known pair actual + full unresolved exposure <= $2. No cache discounts assumed. Release unused reservation ONLY after finite nonnegative consistent usage and actual charged `usage.cost`. Full requested pair token ceilings would cost $0.3821376 at these rates; reservations are not spend.

Price filters cap prompt/completion prices at those exact rates (per million); `provider.only` and `order` are both `["reka/fp8"]`, `allow_fallbacks:false`, `require_parameters:true`, `quantizations:["fp8"]`. Public discovery may be refreshed read-only; drift requires HOLD/new review, never substitution. No extra optional generation flags, sampling changes, parallel_tool_calls, repair plugins, SGLang enable_thinking or separate reasoning token budgets.

## Stop

CANDIDATE_READY, NEEDS_EVIDENCE or normal construction budget exhaustion ends that episode. Pair stops before additional spending on HTTP/network error, missing/invalid usage, unknown exposure, malformed provider response, routing/treatment contradiction, reported token/cost overrun, length truncation, admission failure, local exception, failed containment or deadline. No retry, fallback, automatic rerun, submission, Kaggle launch, new gameplay, cancellation or R10b work. First HTTP failure remains a consumed possibly-billed attempt with unknown spend unless a valid receipt establishes actuals. Known usage survives later local errors. Local timeout retains unresolved in-flight exposure; no remote cancellation claim.

## Invalid if

Prefix/prompt/schema/evaluator initial parity fails; source-format guidance changes between arms; any historical failure/solution hint enters prompt; model/provider changes; missing treatment/usage/routing proof; response pipeline transforms prompt; containment falls back; source/import hashes drift; undeclared request/retry; cap exceeded; freeze promoted after unsuccessful child/deadline; missing approval or unauthorized credential use. No paid call has occurred. The pre-clarification credential availability check is disclosed in `AUTH-ACCESS-NOTE.md`; explicit credential-use approval is still required before execution.

Hosted revision, tokenizer/template, serving implementation, hardware and FP8 details are not hash-equivalent to Kaggle. Ordered singleton episodes cannot separate treatment from provider time variation. Any on failure from requested output exhaustion is recorded as budget-induced truncation, not inability at unconstrained reasoning.

## Result map

- Valid on freeze/repair and off failure: candidate for a separately authorized larger constructor-only study; no efficacy claim or R10 gate change.
- Both valid freezes: record both source/protocol outcomes; no demonstrated advantage; no automatic follow-up.
- Off freeze, on failure/truncation: enabled reasoning did not improve this bounded pilot; retain budget explanation as observation, not causal proof.
- Neither freezes: preserve errors/inspection/submission trajectories; only separately authorized post-run diagnosis, no silent repair/retry.
- Invalid, missing treatment, endpoint/usage ambiguity, timeout or unknown exposure: HOLD interpretation and spending; retain raw evidence and request a scoped next decision.

## Outputs / gates

All new files live under this directory. Preparation: card, dispatch/rulings, source/prefix bindings, engine diff, public discovery, exact candidate inventory, fixture logs and test receipts. Execution: `live/` created once after independent reviewer PASS, Hermes RELEASE and PM credential-use approval. `live/CONSUMED.json` and per-arm consumption markers are never reset. No self-authored approval artifacts.
