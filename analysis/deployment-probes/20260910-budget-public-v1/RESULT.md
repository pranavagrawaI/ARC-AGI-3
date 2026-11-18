# Public Kaggle budget qualification — PASS, operational scope only

Kaggle notebook version **4** completed. The frozen card's operational checks
passed on public LS20, S5I5 and CD82. **No competition submission was made.**

| Game | Solver actions | Completed Actor calls | Levels completed | Game time | Stop |
| --- | ---: | ---: | ---: | ---: | --- |
| LS20 | 1 | 11 | 0 | 92.63 s | Model admission budget |
| S5I5 | 3 | 12 | 0 | 83.84 s | Model admission budget |
| CD82 | 1 | 11 | 0 | 78.54 s | Model admission budget |

Each 75-second admission window stopped new calls. An already admitted call
could finish afterward; these durations stayed within the separate request/tool
and cleanup reserves. The host continued through all three games without the
previous hard decision timeout or an invented action.

## Verification

- All 34 model admissions have matching completed records, all were admitted
  before their deadlines, and all completed before their exhaustion markers.
  No failed lifecycle records, late admissions, summaries or reasoning-on calls.
- Server accounting: 34 successful Solver requests plus 1 warmup equals all
  35 successful chat-completion accesses. No extra/failed request in those counts.
- Whole server log: no OOM, deleted request-state, traceback or HTTP 4xx/5xx
  markers under the audit's explicit patterns. Shutdown reported zero remaining
  requests; lifecycle scans passed with exit code 0 and TERM-only closure.
- Downloaded runtime files, packaged adapter and shared Solver matched the
  frozen contract. Remote notebook cell sources matched the launch snapshot.
- Outer runtime: 584.148 seconds. Last notebook log: 626.328 seconds. These are
  different scopes, not interchangeable setup-time measurements.
- Quota moved from 8.88 to 8.71 GPU hours remaining, about 0.17 hours used
  according to the rounded quota display.

The reproducible read-only [audit.py](audit.py) passed; its output is
[AUDIT.json](AUDIT.json). Full artifacts are preserved in [raw](raw/), including
[game progress](raw/qualification/game-progress.jsonl),
[allocations](raw/qualification/budget-allocations.jsonl),
[outer result](raw/qualification/result.json) and
[server log](raw/qualification/server.log). The [experiment card](EXPERIMENT-CARD.md)
and [launch snapshot](source/) were not changed after launch.

## Interpretation and permitted next action

The budget-exhaustion and shutdown path is now qualified on these three public
sessions with the target model/GPU. This is not solving success: five actions
across 34 Actor calls completed no levels. The public 75-second cap bound every
game, so this run does not validate full-set adaptive allocation, long-game
context behavior, competition gateway execution, or score quality.

Retain version 4 as this exact qualified public candidate. Any competition run
requires separate authorization; none is launched here. No retry was performed.

Notebook: https://www.kaggle.com/code/pranavagrawai/arc3-first-submission-20260910

Notebook SHA256: `b4ff2fc1e8ba3b7c916992252c1e942dff72db24a1f1bbd75014dd09051e1661`.
Runtime contract: `8cb520c9b5070a9e50ed898d57461fbb6530d1a863d57b9a67528ea32374d6c1`.
