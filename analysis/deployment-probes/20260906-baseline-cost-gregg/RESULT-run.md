# baseline-cost-gregg-20260906 — batching collapses at 95k, and the deadline metric reverses

`pranavagrawai/baseline-cost-gregg-20260906`, pushed 16:00:46Z, no queue, terminal **COMPLETE**
16:28:05Z. RTX PRO 6000, model verification `matched`, lifecycle clean. `VERDICT.json`:
`MEASURED_REVIEW_REQUIRED`, `promotion: false`.

**The revision earned its GPU time.** It independently replicated the original's captured-set
numbers and then produced two findings the original package could not have surfaced.

## 1. Independent replication

Two separate kernels, same captured four-prompt set:

| concurrency | original wall | gregg wall |
| --- | --- | --- |
| 1 | 30.15 / 30.01 s | 30.14 / 30.03 s |
| 2 | 18.58 / 17.59 s | 18.60 / 17.63 s |
| 4 | 10.86 / 10.93 s | 10.77 / 10.91 s |

Agreement within ~1 %. The batching effect at Actor prompt lengths is reproducible across runs, not
a single-kernel artifact.

## 2. The batching win does not survive long context

The 95,000-token mechanical set, same protocol:

| concurrency | captured set wall | 95k set wall | 95k aggregate tok/s |
| --- | --- | --- | --- |
| 1 | 30.14 s | 133.65 s | 7.66 |
| 2 | 18.60 s | 113.61 s | 9.01 |
| 4 | **10.77 s (−64 %)** | **103.83 s (−22 %)** | 9.86 |

Concurrency 4 buys a 64 % elapsed-time reduction at captured lengths but only 22 % at 95k. Server
overlap confirms why: mean `num_running_reqs` in the 95k C4 window is 1.68 against 2.76 for the
captured C4 window. Most 95k time is spent in prefill, which does not overlap the way decode does —
consistent with the original run's finding that prefill is 70.8 % of a 95k request.

## 3. Queue-inclusive deadline accounting reverses the 95k verdict

This is the contrast the revision was built for. Counting from the common workload-ready instant
rather than from dispatch:

| set | C | median ready→completed | max | median executor queue |
| --- | --- | --- | --- | --- |
| captured | 1 | 19.08 s | 30.14 s | 11.29 s |
| captured | 2 | 14.44 s | 18.60 s | 5.14 s |
| captured | 4 | **10.76 s** | 10.76 s | 0.00 s |
| 95k | 1 | **84.84 s** | 133.65 s | 51.49 s |
| 95k | 2 | 86.07 s | 113.61 s | 29.27 s |
| 95k | 4 | **103.83 s** | 103.83 s | 0.01 s |

At captured lengths concurrency improves both throughput and per-request time-from-ready — a real
win. **At 95k it does not:** median ready-to-completion degrades from 84.84 s to 103.83 s going C1 →
C4, while aggregate wall time improves. Serial execution lets early requests finish early; C4 makes
all four finish together, later.

That is precisely the card's caution that "timing-only improvement with unacceptable deadlines is
not a useful win." An optimization selected on aggregate throughput alone would have been chosen
here on evidence that reverses under the metric that matters for a deadline.

All 24 requests completed within the 180 s diagnostic budget in every configuration, so that budget
did not discriminate; the median/max ready-to-completion figures are what carry the signal.

## Validity and limits

Server overlap observed in every set (`num_running_reqs` reaching exactly 1/2/4 in the matching
windows, 2,122 metric samples). Lifecycle clean: `status passed`, `containment_passed true`,
`root_returncode 0`, `forbidden_count 0`.

- **No promotion, no bottleneck diagnosis, no backend selected.** MTP stays parked.
- **Isolated HTTP concurrency, not host admission qualification.** This does not qualify the
  packaged host, ownership, result routing or real game deadlines.
- **The 95k set is mechanical** prefix repetition, not four natural Actor contexts; the captured
  set is one game replayed, so it shares more prefix than independent games would.
- **The 180 s common-ready measure is diagnostic**, not a game deadline.
- **Traces establish no saturation** without hardware counters.

## What follows

If a single intervention is chosen from this, the evidence points at the prefill/cache boundary
rather than at concurrency: prefill is 70.8 % of a 95k request and 28.9 % of a real 21k Actor
request, and historical prefix reuse already retired 86.6 % of a successor's prompt in the
continuation control. Concurrency's benefit is real but confined to the shorter captured lengths and
does not hold under deadline accounting at long context. Selecting it is still a separate decision
requiring host and deadline qualification, which this run does not provide.
