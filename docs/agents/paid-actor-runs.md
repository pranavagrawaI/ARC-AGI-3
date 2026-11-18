# Paid actor runs

Lessons from arc-agi-3's paid actor runs (`analysis/matched-actor-*`, `analysis/parity-compat-*`, `analysis/production-actor-paid-shadow-v1`). Recheck provider facts at use time, since endpoints and pricing change.

For paid tool loops, distinguish current-request reservations from validated actual usage and reserve a final submission/evidence phase before exploratory work can consume the budget. Missing usage must stop further spending, not count as zero.

Before pinning a paid multi-provider request, compare the complete outgoing parameter set against the selected endpoint's live metadata, including optional flags such as `parallel_tool_calls`. Advertising tools and reasoning support alone does not establish compatibility with strict `require_parameters` routing. Preserve discovery evidence and distinguish an advertised active endpoint from a provider that actually returned a completion.

Qualify the advertised action horizon against cumulative prompt/completion admission and maximum unresolved-request exposure, not only per-call limits. Replay real prior payloads and the actual frozen model's query/retrieval outputs, including accumulated history and maximum notes, before release. A lossless codec must round-trip through the real actor boundary for both arms; common compression does not prove the model interprets it equally well. Distinguish budget eligibility for bounded inputs from a guarantee that every future observation fits.

Use a per-request absolute deadline below the phase deadline so one stalled response cannot consume the entire actor phase; nested timers must restore remaining outer time, not restart it. Keep absolute-deadline enforcement active while provider I/O blocks; a socket inactivity timeout does not bound a trickling response. Before launch, test both slow successful bodies and slow HTTP-error bodies through the real supervisor and transport. On cutoff, reap the owned worker, retain unresolved exposure, forbid late publication and further calls, and do not claim provider cancellation.

Before launch, exercise the audit on both a completed response and a first-request HTTP failure. Record latency in failure paths as well as success paths. An empty successful-response set establishes neither zero billed tokens nor zero cost; report these as unknown without a receipt.

Validate multi-turn provider contracts with provider-shaped response fixtures, not only idealized tool calls. Preserve raw responses, but project response-only metadata (for example a tool call `index`) out of outbound request messages at a reviewed boundary; otherwise an overstrict request validator can accept the first paid request and block continuation after its tool executes. Exercise the real constructor-to-wire continuation with that fixture before freeze.

Distinguish a local pre-send contract failure from an attempted request with no receipt. Reconcile known usage from completed raw receipts when every actual attempt completed; do not relabel it unknown because a later local exception bypassed summary persistence. Persist construction progress and its tool transcript incrementally so local failures retain accounting and diagnostic evidence.

Before attributing repeated tool failures to model reasoning, compare the actual outbound nested schema with the host validator and prompt. Exercise prior malformed arguments through the real constructor boundary, check precise type/key/nesting feedback, and verify a subsequent valid call reaches sealed evaluation without coercion. Document unavoidable schema/parser differences rather than claiming universal parity. A schema-plus-feedback repair is a joint intervention; first-try success does not exercise repair feedback.

Separate accepted executable construction from predictive support. Inspect frozen source, declared commitments, evaluator targets and scoring: copying the current observation with explanatory notes can pass ABI replay without making a tested dynamics prediction. Preserve this distinction even when both reasoning treatments freeze successfully.

Separate recoverable actor tool mistakes from integrity/provider failures before a utility run. Exercise an out-of-range history lookup through the real actor loop and predeclare bounded error feedback or fail-fast behavior. A static schema can admit a lookup that violates the current executed prefix; aborting the whole pair on that ordinary error can prevent any utility comparison despite complete receipts and valid containment. Do not silently relax the policy after observing the failure.
