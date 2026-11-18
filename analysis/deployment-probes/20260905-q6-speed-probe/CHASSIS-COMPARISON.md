# Three chassis on the same Kaggle FP8 route, 2026-09-05

All three ran the same model, SGLang build and hardware on the same day. The two reference arms
played ls20/cd82/s5i5 concurrently against one server; the q6 probe played ls20 alone.

| | q6 speed probe | tycho-upstream | nooa-upstream |
| --- | --- | --- | --- |
| runtime | 28 m (capped) | 7 h 05 m (timed out) | 2 h 51 m (worker timeout) |
| games | ls20 only | 3 concurrent | 3 concurrent |
| terminal | action cap reached | `rc=124`, `timed_out: true` | `TimeoutError` on decision |
| ls20 actions | **10** | 18 | 2 |
| ls20 grid changes | **9** | 17 | 1 |
| levels completed | 0 | 0 | 0 |
| solver decisions | 6 | not recorded | 5 (all games) |

Per environment action on ls20, dividing the reference arms by three for contention:

| chassis | raw s/action | contention-adjusted |
| --- | --- | --- |
| q6 | **133** | 133 (already solo) |
| tycho-upstream | 1417 | ~472 |
| nooa-upstream | 4946 | ~1649 |

q6 is the fastest chassis on this route by a wide margin, roughly 3.5x tycho's adjusted rate and an
order of magnitude past nooa-upstream. Tycho is effective per action (17 grid changes in 18) even
though it is slow; nooa-upstream is both slow and ineffective (1 change in 3 games).

**No chassis completed a level on any game**, with up to seven hours of play. Level progress is not
the near-term measure to size an arm against.

A launcher timeout is survivable evidence, not necessarily a crash: tycho hit its own limit, was
SIGTERMed, and still wrote per-game progress. The earlier worry that a timeout necessarily destroys
artifacts is not borne out for that exit path, though `result.json` still reports `status: invalid`.

## A third session schema exists

`reference-arm-sessions-v1`, with `total_actor_calls: null` and no `model_evidence`, and
`submitted_batches: 0` for every game. So the route now has four session schemas in play:

| producer | schema | Actor-call evidence |
| --- | --- | --- |
| Tycho public25 probe | `tycho-nooa-public-solver-sessions-v1` | yes |
| q6 join (repaired) | `nooa-join-sessions-v4` | yes |
| NOOA join (previous) | `nooa-join-sessions-v3` | no |
| reference arms | `reference-arm-sessions-v1` | no |

The repaired reconciler accepts the first two and rejects the rest, which is correct — the others
carry nothing to reconcile. Neither reference arm reached reconciliation this time (one died on a
worker timeout, the other on the launcher limit), so whether they would fail there is untested.
