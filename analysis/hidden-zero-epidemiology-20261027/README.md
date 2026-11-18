# Hidden-run zeros are a platform load-time event, and a keep-alive cell removes the mechanism (2026-10-27)

## 1. D17 changes what the zeros mean

`submissions.json` (`fetch.py`, a full API dump) holds three COMPLETE zeros, each with an identical **3,411 B** output: D9 F1 (10-18),
D12 F1x (10-21) and **D17 (10-26)**. D17 is the unchanged control, scriptVersionId 354278759. That same version scored 23.87, 29.44,
24.97, 29.38 and 31.31 between 10-01 and 10-17. D16 (F1x+REAP, 10-25) scored a partial 6.52.

- The F1-lineage association behind the quarantine (2/3 vs 0/10, grouping chosen after the fact) no longer holds. The
  control now zeros too.
- Every zero falls on or after 10-18, and none before (0/12 hidden runs, 09-19..10-17).
- Kaggle discussion 751598 (`../hidden-draws-20261010/d16-f1x-reap/diagnosis-20261026/`) has more than 15 teams reporting the same 3,411 B
  COMPLETE zero on unchanged dfranzen-lineage code, starting around 10-17. Kaggle staff acknowledge regional dataset-cache
  instability and mention rescores.

## 2. Our own logs show the slowdown (`load_series.py`, `load_series.json`)

185 GB model precache: **220–417 s** on every retained run from 09-30 to 10-20. Then **2,010 s** on 10-22 (m2_guard fault
screen) and **895 s** on 10-25 (c16 screen-2). READY was 463–676 s through 10-20, already close to the 720 s release.
Teams in 751598 report 35–63 min loads.

## 3. The mechanism, reproduced (`run_competition_idle.py`, `run_matrix.sh`)

The stock notebook releases the benchmark at 12 min with one warm-up RESET. The gateway closes a scorecard that has seen
no action for 15 min (reported in 751598/750173). Every later action gets HTTP 400, which is the fast 4xx path that
`../f1x-hidden-zero-diagnosis-20261021/faults/FAULTS.md` had predicted from D12's ≤2.2 h completion. The CPU harness
here is the f1x-diagnosis harness, plus a gateway idle close and execution of the keep-alive cell. It ran the control
notebook's own cells at 1/10 time scale: release 72 s, idle 90 s, first-request grace 90 s, load 240 s (= 40 min) vs
20 s, per-game cap 600 s.

| run | scorecard | actions accepted / rejected | games acted |
|---|---|---|---|
| stock, slow load | **closed at 240 s** | 111 / 9,889 | 1 |
| keep-alive, slow load | open | 1,445 / 0 | 23 |
| stock, fast load | open | 2,132 / 0 | 33 |
| keep-alive, fast load | open | 2,136 / 0 | 33 |

Fake actions are random, so levels are 0 everywhere and are not a gameplay signal. Accepted actions and games acted are the
signal. In the slow keep-alive run, the gap to the fast runs is the load time itself.

## 4. The fix (`build_keepalive.py`)

The builder adds one cell before the run cell and leaves every parent cell byte-identical. Each game that holds a stream
slot waits before its first analyzer turn until `/health` returns 200. While games wait, one engine RESET goes out every
360 s through the stock warm-up path. The hold is capped at 150 min from notebook start. The 12-min release and the
per-game clock are unchanged, so the cell cannot push the run past the 9 h limit. When the load is fast, the cell is a
no-op: 0 RESETs. Cost under a slow load: about one RESET per 6 min of load, spread across held games, roughly one action
on level 1 of a few games.

Not proven: the real gateway's idle rule, which we know only from other teams' reports, and whether a live Kaggle rerun
behaves like the local gateway. One team ran a hold+keepalive variant on 10-23 and got past the zero, then hit a Kaggle
"system error" ERROR at about 7.5–8 h, cause unknown. Kaggle has refunded ERROR slots.

## 5. Consequences

- Draws since 10-18 are confounded by platform state. D16's 6.52 and possibly D11's 23.32 are not arm evidence.
- The F1/F1x quarantine rests on a refuted premise. F1x's valid hidden record is one draw, 30.88.
- The raw-control fallback still has the 12+15 min trap.
- `RESCORE-REQUEST-DRAFT.md` drafts a rescore request for 751598. It has not been posted.

## 6. GPU smoke (pushed 2026-10-26 23:53Z, kernel pranavagrawai/m2-f1x-keepalive v1, id 141067143)

This Save & Run is both the smoke and the submittable version. Pass rule, fixed before any output was seen:
COMPLETE; RTX PRO 6000; "M2 KEEPALIVE armed" printed; no crashed games and no keep-alive exception; if the server was not
READY by the 720 s release, keep-alive RESETs and a "ready" line appear. No-harm: public-25 levels at 60 min after the
benchmark starts ≥ 69 (F1x screen-2 got 76; the pre-registered 1 sd is 7). On a pass, queue it for the next free draw. This
is the same F1x code as the 30.88 draw plus the keep-alive cell.

**Result (COMPLETE 01:08Z, 75 min): PASS.** RTX PRO 6000 Blackwell. The server was not ready at the 720 s release (stock
DEADLINE line), so the hold engaged on real hardware: one keep-alive RESET (bp35, t=724 s), 10 games held, released at
READY t=785 s. No tracebacks, states {cancelled, won}. Levels@60 = 72 (floor 69; F1x screen-2 76, inside its 9-level
replicate spread). Preload NameErrors 0, 16K cutoff in effect. The "ready" line printed several times from a harmless
race between threads that see READY at once; it was left alone so the smoked bytes stay the drawn bytes. Queued as D18
(2026-10-27) in `../hidden-draws-20261010/queue.json`.

## 7. D18 read rule (submitted 2026-10-27 01:31Z, written before the result)

- COMPLETE 0.00 with 3,411 B, or completion within about 2 h: the keep-alive did not stop the hidden zero. Then the idle-close
  reading is wrong or incomplete; go back to the mechanism before spending another draw on it.
- A normal score (≥ 20) after about 9 h: consistent with the fix, but not proof. That day's load may simply have been fast.
  Kaggle shows us no hidden log. Only repeated normal scores while other teams still report zeros would count as evidence.
- Partial (0 < score < 20) or ERROR: a different platform fault, like D16 or the other team's 7.5 h ERROR. ERROR refunds
  the slot.
