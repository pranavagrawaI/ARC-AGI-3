# Matched Actor Treatment V1-Pilot

## Incumbent and question

Native Actor with explicit bounded Python is the production incumbent. CodeAct
advances only by purchasing measurable correctness or resource value on frozen
ARC judgments; eventual protocol compatibility is insufficient.

## Sample and intervention

The pilot is a committed six-packet sampling layer over the untouched 12-packet
V1 corpus: two computation-unnecessary, two computation-useful, and two
composition cases. The schedule alternates Native-first and CodeAct-first. Both
arms retain the V1 prompts, schemas, evidence, shared pure computation, no-write
authority, and no ARC-action authority. Each invocation has at most six provider
calls, 16,384 input tokens per call, 4,096 output tokens per call, and 120 seconds.

## Measurements

Preserve factored correctness and the full V1 resource/milestone vector. Derive
control-language tax only where both milestones exist:
`terminal_result_accepted - candidate_answer_formed`. A censored arm does not
invalidate a clean counterpart, but an incomplete pair cannot count toward
advancement.

## Advancement

All six pairs must be complete. CodeAct advances to the 12-packet treatment only
if it wins at least two packets, loses no more packets than it wins, and is not
less correct overall; or if correctness is equal and its total provider calls or
dollar cost are at least 20% lower. Both arms below four correct packets means
the model is unreliable. Every other complete outcome selects Native.

Provider failure, route drift, missing usage, manifest drift, authority drift,
or unaccounted dispatch stops the run. No adaptive reruns.

A CodeAct failure before target evidence disclosure or useful computation is an
interface/control failure and blocks the full treatment. Native being more
correct selects Native. Equal correctness with materially higher CodeAct cost
also selects Native.
