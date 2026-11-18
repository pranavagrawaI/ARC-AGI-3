# Result — A and B valid; factorial incomplete

## Verdict

The repaired request-purpose boundary passed. Arms A and B each completed two valid decisions and two real environment transitions with clean reasoning-off and request accounting. Arm C then stopped on a DNS failure during its second Actor request, so C and D are untested and the four-arm comparison is incomplete.

The only completed treatment comparison is unfavorable to the measured-evidence capsule in this single sample: B used 23 physical requests and 203.82 seconds versus A's 11 requests and 88.18 seconds. This does not establish that evidence generally hurts—the stochastic arms chose different first actions—but it is sufficient not to promote B from this run.

## Completed-arm measurements

| Measure | A | B |
|---|---:|---:|
| Decisions / transitions | 2 / 2 | 2 / 2 |
| Physical requests | 11 | 23 |
| Actor / summarizer requests | 10 / 1 | 19 / 4 |
| Arm wall | 88.18 s | 203.82 s |
| Reported prompt tokens | 180,484 | 388,363 |
| Reported cached prompt tokens | 119,168 | 244,608 |
| Completion tokens | 4,516 | 11,723 |
| Provider cost | $0.0559496 | $0.1293622 |
| Reported reasoning tokens | 0 | 0 |
| Levels | 0 | 0 |

B's fourth summarizer request completed upstream during cleanup and was recorded as completed-not-delivered. Its 23 physical responses reconcile to 22 worker model-call records plus that one explicitly retained delivery outcome. Because background summarization overlaps Actor work, summed model-call wall exceeds arm wall and must not be interpreted as serial elapsed time.

## Behavioral inspection

A chose `UP`, measured the resulting 52-cell change, then chose `LEFT` to test whether the visible bands represented camera motion. B chose `RIGHT`, received a capsule containing the exact 52-cell transition, changed bounding box and color changes, then chose `LEFT` to test its revised claim that the moving 9/c region was the controlled object. Thus B did use the preceding measured evidence in its second decision; it did not merely mention the capsule.

Both arms nevertheless built incorrect or incomplete movement models and reached no level. B's capsule worked as an information interface but did not reduce investigation. It approximately doubled Actor calls and triggered four summarizations instead of one. Python execution itself was negligible (0.09 s A, 0.14 s B); the additional cost came from model calls and their longer histories, not host-side evidence computation.

Neither arm wrote an executable retained procedure. The run therefore provides no evidence yet for the intended fast path where discovery eliminates future inference.

## C failure and scope

C's first Actor request completed. Its second request terminated with `URLError: Temporary failure in name resolution` before an upstream response. Admission was closed, the worker was aborted, and D was not started, as required by the card. This is an upstream transport failure rather than evidence about the demonstration.

Total reported provider cost was $0.1894816. There was no forwarding retry or relaunch.

## Decision

- Do not promote B: it was functional but substantially more expensive than A in the only matched completed sample.
- Make no claim about C, D, or the demonstration treatment.
- Retain reasoning-off, bounded context, purpose-separated summarization, and cost instrumentation; those mechanisms qualified.
- The next discriminating run, if authorized, should compare A versus C/D with randomized or counterbalanced order and exclude B unless the capsule is materially simplified. It needs a fresh card because this run reached its technical stop.
