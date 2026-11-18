# Independent review — budget-aware public Kaggle qualification v1

Reviewer: independent Claude agent (Superset), 2026-09-10. Read-only.
Task ID: `budget-public-claims-review`. Brief: [REQUEST.md](REQUEST.md).

> **See [Addendum](#addendum--corrections-to-this-review-2026-09-10) at the end.** Five items below are
> corrected there and the addendum supersedes them: recommendation 2 (local prompt-growth run),
> recommendation 3 (dose-response), the "model loading is 148.1 s" phrasing in claim 4, the
> unconditional wording of F2, the scope of F1/F3, and an arithmetic magnitude in claim 6.

Scope of authorship: this file plus [recheck.py](recheck.py) and
[simulate_allocator.py](simulate_allocator.py). No source, notebook, card, log or
existing report was modified. No Kaggle call, upload, submission or paid model call.

**Bottom line.** The numbers hold up. I reproduced every figure in claims 2–6 from
the raw artifacts, and the operational PASS survives an error scan wider than the
coordinator's own. Three things need correcting or qualifying, in order of
consequence: the solver description in claim 7 says the prediction gate *checked*
predictions when the raw records show it returned `prediction not committed` on
every batch in all three games; claim 6's "228.5 s/game" hides the fact that the
allocator is strongly back-loaded (first game ≈91.9 s of model admission, last game
≈750 s); and the "version 4" identity in claim 1 is not establishable from the
preserved evidence, only from the coordinator's own narrative. Nothing here
invalidates the run or the frozen card.

---

## What I ran vs. what I inspected

| | |
| --- | --- |
| Ran | `audit.py` (reproduced `AUDIT.json` byte-identically, exit 0) |
| Ran | `claude-review/recheck.py` — independent recomputation of claims 2, 3, 4, 6 |
| Ran | `claude-review/simulate_allocator.py` — re-implementation of `play.py:26-40` at 110 games |
| Ran | `pytest submission-20260910-v5/{test_submission,test_budget_accounting}.py -q` → **23 passed in 84.11s** (report says 23 in 94.68 s; count matches, wall time varies) |
| Ran | widened error scans over `server.log`, `teardown.log`, `events.log`, all `worker-std{out,err}.log` |
| Inspected | `play.py`, `build.py`, `exact_request_outer_runtime.py`, `headless.py`, `shared_local_solver.py`, `nooa_worker.py`, `nooa_public_probe.py`, `adapter.py`, `chassis/{solver_agent,arc_llm,contingent_controller}.py` |
| Not possible here | Kaggle kernel version ordinal, competition submission state, GPU quota display — all off-machine, and no Kaggle call is authorized |

---

## Claim-by-claim

### 1. Version 4, public-only, no submission, frozen PASS valid — **conditional**

Verified independently:

- The run's *source identity* is sound. `raw/nooa-join/CHASSIS.json` and each
  session's `worker-metadata.json` carry `contract_sha256`
  `8cb520c9…74d6c1`, matching `source/BUILD.json`; notebook SHA256
  `b4ff2fc1…51661` matches `deployment/kaggle/submission-20260910-v5/submission.ipynb`
  byte-for-byte (I re-hashed both).
- The launch command preserved in `raw/qualification/lifecycle-target-launch.json:7`
  is exactly the non-rerun branch of `build.py:81-87`
  (`--launcher-timeout-s 4800`), and `budget-allocations.jsonl` shows
  `admission_seconds 75.0`, i.e. `SUBMISSION_ADMISSION_CAP_SECONDS='75'`
  (`build.py:72`). The evidence is internally consistent with *this* notebook.
- `EXPERIMENT-CARD.md` mtime 05:12 IST; the run's `started_at` is
  `2026-09-09T23:43:08.887964+00:00` = 05:13:08 IST. The card predates the launch.
  `source/` files predate it too. Consistent with "not changed after launch".
- Operational PASS under the literal card: no OOM, no deleted request-state, no
  traceback, no non-200 HTTP anywhere in `server.log`; **all 38 HTTP responses in
  the log are 200** (35 chat/completions, plus `/v1/models`, `/model_info`,
  `/freeze_gc`). Zero `failed` lifecycle events across 34 request pairs. No
  `ABORT` marker anywhere under `raw/`. `Gracefully exiting… Remaining number of
  requests 0` at `server.log:31177`. Worker stdout/stderr are 0 bytes in all three
  sessions. No recovered failure is being excused, because there was no failure.

**Not establishable:** nothing in `raw/`, `terminal-log/`, `session-evidence/` or
`remote-source/` carries a Kaggle kernel version number. `remote-source/…ipynb`
is a *source* readback (no outputs), so it attests that the kernel's current
source matches the launch snapshot — not that the archived run was ordinal 4.
`analysis/deployment-probes/20260910-first-submission/STATUS.md` itself warns that
the installed CLI does not pass a version in the output request. I report "version
4" as the coordinator's assertion, corroborated only by its own push history.
Likewise, "no competition submission" and the 8.88→8.71 GPU-hour quota are
off-machine facts I cannot check; note that `build.py:92` does write
`/kaggle/working/submission.parquet` on the public path (it is present in `raw/`),
which is an artifact, not a submission.

### 2. Session times, actions, calls, request reconciliation — **verified**

Recomputed from `game-progress.jsonl` and each session's `ipc/`:

| Game | wall (s) | actions | completed Actor calls | levels |
| --- | ---: | ---: | ---: | ---: |
| LS20 | 92.62774050200005 | 1 | 11 | 0 |
| S5I5 | 83.83993034100001 | 3 | 12 | 0 |
| CD82 | 78.54498956999987 | 1 | 11 | 0 |

All three `stop_reason` = `model_admission_budget`; all three
`budget_exhausted.json` have `active_requests: 0` and `at_monotonic ≥
admission_deadline`. 34 completed calls; **34 + 1 warmup = 35**, and I confirmed
this against the raw log rather than `result.json`: `grep -c 'v1/chat/completions'
server.log` → 35, all 200. All 34 calls have `request_purpose: "actor"`,
`summarizer: false`, `reasoning_present: false`, `reasoning_tokens: 0`,
`finish_reason: "tool_calls"`. Admission margins and tails:

| Game | last admit before deadline | last completion past deadline | max single request |
| --- | ---: | ---: | ---: |
| LS20 | 11.346 s | 17.208 s | 28.555 s |
| S5I5 | **1.008 s** | 8.420 s | 13.539 s |
| CD82 | 4.246 s | 3.125 s | 16.341 s |

All tails sit far inside the 90 s request reserve, so RESULT.md is right. Worth
naming the mechanism though: S5I5 admitted a call 1.008 s before its deadline.
The guard (`solver_agent.py:772-775`) admits on `started < deadline` with no
look-ahead, so the reserve — not the guard — is what bounds the tail. The reserve
was sized for it; the observed run simply never tested it near the limit.

### 3. Model-call wall times and distribution — **verified, with the convention named**

| | LS20 | S5I5 | CD82 | total |
| --- | ---: | ---: | ---: | ---: |
| Σ `wall_seconds` | 88.40476135200004 | 79.29067918999999 | 74.56394400800013 | **242.25938455000016** |

242.25938455 / 255.019576933 = **0.94996** → "~95%" is right.
n=34: mean **7.125276016**, median **5.643897107**, max **28.554694850**,
min 1.658690320. Output tokens **8539**; reasoning tokens 0; prompt range
**13165–29906** (total prompt 659223).

**Percentile convention (claim 3 asked for it):** p90 = 13.729653012 is
*linear interpolation on rank* `0.9·(n−1)` — numpy's default and
`statistics.quantiles(w, n=10, method='inclusive')[-1]`. The alternatives differ:
nearest-rank gives **13.811175075**, Excel-exclusive gives **13.877569512**.
State the convention when quoting it. With n=34 the p90 is pinned by three
adjacent order statistics (13.539, 13.811, 13.944) with a lone 28.555 s outlier
above them — treat it as descriptive, not as a latency bound.

### 4. Clock boundaries and the 371.3 s residual — **verified, and the attribution is correct**; call it an approximation

Notebook last log **626.327837106 s**, outer **584.148 s** (I confirmed against
`ended_at − started_at` = 584.14753 s), gameplay **255.019576933 s**.
626.327837106 − 255.019576933 = **371.308260173 s**.

The coordinator's key assertion — this is combined non-game overhead, **not** all
model loading — is correct, and `raw/qualification/events.log` lets me quantify it.
Anchoring the notebook clock on the "verifying immutable model files" line
(notebook 29.328 s = 23:43:09.14 UTC):

| Segment | notebook s | duration | share of 371.3 |
| --- | --- | ---: | ---: |
| kernel start → outer start (wheelhouse verify, venv, 77 pkgs) | 0 → 29.08 | 29.08 | 7.8% |
| immutable model-**file integrity verification** | 29.33 → 204.95 | **175.62** | 47.3% |
| SGLang launch + warmup (`warmup complete after 148.126s`) | 204.95 → 353.08 | **148.13** | 39.9% |
| *(gameplay — excluded)* | 353.08 → 608.10 | *255.02* | — |
| accounting, teardown, result print | 608.10 → 626.33 | 18.23 | 4.9% |

Sum 371.06 s; the 0.25 s gap is unattributed slack. So model *loading* is 148.1 s —
40% of the residual — and the single largest item is hashing the model files, not
loading them. Two consequences the report should carry: that 175.6 s is a
one-time cost that a longer run amortizes, and it is a plausible target for
reduction independent of anything else.

**Answer to "approximation or total":** approximation. 371.308260173 subtracts a
launcher-internal monotonic clock (255.019576933) from a notebook-relative clock
(626.327837106); the two agree only because the launcher window nests inside the
notebook window. It also excludes whatever the kernel does after its last log line,
and the notebook's own first log arrives at 4.088 s. Quote it as **≈371.3 s**, and
prefer the decomposition above to the single number. `584.148 − 255.020 = 329.128 s`
is the tighter, single-clock statement of overhead *inside* the outer runtime.

### 5. Competition launcher budget — **verified from the invoked path, not defaults**

- `build.py:85` — `'--launcher-timeout-s', '25200' if rerun else '4800'`.
- `exact_request_outer_runtime.py:2189-2190` — the probe is invoked with
  `--timeout-s str(args.launcher_timeout_s - 60)` → **25140 s = 6h59m00s**.
- `exact_request_outer_runtime.py:2194` — that subprocess carries
  `timeout=args.launcher_timeout_s` = 25200 s.
- `build.py:88` — the notebook's outer `subprocess.run(…, timeout=27600 if rerun else 6300)`.

The `−60` rule is not just source-reading: the public run passed
`--launcher-timeout-s 4800` (`lifecycle-target-launch.json:7`) and the first
allocation records `remaining_seconds: 4739.99985163`
(`budget-allocations.jsonl:1`), i.e. `timeout_s` = 4740 = 4800 − 60. Confirmed
empirically. 25200 s = 7h00m of a 9h (32400 s) assumption — **the launcher does not
exploit it**, and the outer 27600 s guard is 7h40m.

### 6. The 110-game arithmetic illustrations — **verified as arithmetic; one framing is materially misleading**

Both figures reproduce, once the rate convention is pinned:

| | coordinator | mine | note |
| --- | ---: | ---: | --- |
| 25140 / 110 | 228.545 | 228.5454545 | ✔ |
| total actions | ~492.903 | 492.9034 | uses rate = 5 / **255.019576933** (launcher clock) |
| per game | ~4.48 | 4.4809 | ✔ |
| (32400 − 371.30826)/110 | 291.170 | 291.1699249 | ✔ |
| total actions | ~627.965 | 627.9650 | ✔ |
| per game | ~5.71 | 5.7088 | ✔ |

Using the sum of the three game clocks (255.012660) instead gives 492.9167 /
4.4811 — a 0.01 action/game difference. Either is defensible; say which.
"23 of 34 calls preceded first actions (6+8+9)" is **exact**: turn/step-0 call
counts are LS20 6, S5I5 8, CD82 9 = 23, and these do include the call that emitted
each first action.

**The framing problem (finding F2 below).** Labelling them "illustrations, not
forecasts" is necessary but not sufficient, because a reader will take 228.5 s/game
as *what each game gets*. It is not. I re-implemented `admission_budget`
(`play.py:26-40`) and the per-game loop (`play.py:84-102`) at 110 games,
T = 25140, request timeout 90 (`simulate_allocator.py`):

| overhead/game | 1st game admission | 10th | 50th | 110th | mean admission | mean game wall | unused |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3.54 s | 91.909 | 103.2 | 169.9 | **785.1** | 222.2 | 225.71 | 1.24% |
| 10.0 s | 91.909 | 102.6 | 166.1 | **751.1** | 215.8 | 225.77 | 1.21% |
| 17.63 s | 91.909 | 102.0 | 161.5 | **710.8** | 208.2 | 225.84 | 1.18% |

(the three overheads are the observed per-game `wall − 75 s`: 17.628 / 8.840 / 3.545.)

Two things follow. First, the **mean** game wall converges to ~225.8 s, so 228.5 is a
fair *average* and the pool is ~99% consumed — the arithmetic is not misleading
about totals. Second, the schedule is **harmonically back-loaded**: the 135 s of
per-game reserves (90 request + 15 tool + 30 cleanup) are never spent, so each
game returns ≈135/(N−i) s to the pool and the allowance grows without bound toward
the tail. Game 1 gets **91.9 s** of model admission; game 110 gets **~750 s**, an
8× spread. The v5 README (`README.md:26-28`) already states the 91.9 s figure, so
this is consistent with the design — but a flat "4.48 actions/game" reads as
uniformity that does not exist, and if a session is cut short the early games are
the starved ones.

On the other listed concerns:

- **Quota vs session runtime.** Correctly distinguished elsewhere; the 25140 s is a
  *play* window inside a 25200 s launcher guard inside a 27600 s subprocess guard,
  all inside whatever Kaggle session cap the competition rerun actually grants.
  The illustrations say nothing about quota and shouldn't.
- **Per-game reserves.** Not deducted in the illustration. For early games the
  model-admission window is `wall − 135 s`; at game 1 that is 91.9 s of 226.9 s.
- **History growth.** Genuinely unresolved and the reason no forecast is licensed.
  Prompt tokens grew **+786 / +1371 / +1517 tokens per call** (CD82/LS20/S5I5 means)
  and reached 29,906 in 12 calls. `headless.py:126-130` sets
  `TruncationConfig(max_context_tokens=20_000)` against `ARC_LLM_CONTEXT_WINDOW=262144`
  — and observed prompts already exceed 20,000, so that setting plainly does not
  bound the total prompt. Whether prompts plateau or grow linearly over a 600 s
  window is unknown, and it drives both latency and whether calls survive at all.
- **Startup amortization.** Cuts the other way and is not in the illustration.
  23 of 34 calls (68%) went to orientation before the first action; over a
  92 s window that is most of the budget, over a 750 s window it is ~12%. Marginal
  post-orientation cost was 1.33–5 calls per action. So the linear extrapolation
  is pessimistic on amortization and optimistic on context growth, and the two
  errors are not known to cancel.

**Verdict:** useful as bounded arithmetic, provided (a) "228.5 s/game" is stated as
a *mean* with the 92→750 s spread named, (b) the admission window is distinguished
from the wall share, (c) the labelling as illustration survives into any downstream
quotation. No score forecast is licensed, and I agree with that.

### 7. Solver description — **verified with two corrections**

Confirmed from source and from this run's records:

| Element | Status |
| --- | --- |
| NOOA `MdArcSolverAgent` via SGLang, Qwen3.8-27B-FP8 | ✔ `worker-metadata.json` `agent: nooa.MdArcSolverAgent`; model source `jakobbrggen/qwen3-8-27b-fp8-hf-snapshot` |
| sequential games | ✔ `play.py:84` single for-loop; `shared_local_solver.py:49` BoundedSemaphore(1) |
| grid/text default, not image vision | ✔ `headless.py:53` `--visual` default `"off"`; `show(` appears 0 times in all 50 tool payloads |
| CodeAct Python/NumPy | ✔ all 50 tool calls are `execute_python`; `np.` used 33× across the three games |
| only `submit_actions` advances gameplay | ✔ `submit_actions` appears 5× in tool payloads = 5 executed actions |
| routine reasoning off | ✔ `ARC_LLM_EXTRA_BODY` disables both switches (`arc_llm.py:98-116`); `headless.py` passes no `effort_ladder`, so `_install_effort_ladder` returns early (`solver_agent.py:859-863`); 34/34 calls `reasoning_tokens 0` |
| background summarization disabled | ✔ `headless.py:154` `enable_summarizer=False`; 0 summarizer calls |
| each model call guarded by the admission deadline | ✔ `solver_agent.py:761-786`; `ARC_GAME_MODEL_ADMISSION_DEADLINE` is in `WORKER_ENVIRONMENT_KEYS` (`shared_local_solver.py:60`), and 34 admitted/34 completed/0 failed prove it fired |

**Correction 7a — "observations/predictions checked; batches gated on predictive
agreement" overstates this run.** The gate ran on every batch and returned
*unresolved* every time. All five `step_checks.jsonl` rows across the three
sessions read `"status": "unresolved", "reason": "ValueError: current prediction is
not committed"`, and the matching `batch_stops.jsonl` rows read
`"reason": "prediction_unresolved", "skipped_actions": 0`. No prediction was ever
committed, so no observation was ever compared against one. Because every batch
contained exactly one action, nothing was skipped and behaviour was unaffected —
but the mechanism was *not* exercised as described. Correct wording: *the
prediction gate is installed and fired on every batch; in this run it resolved to
"unresolved" each time because no prediction was committed, so no batch was
truncated.* This also means the gate's discriminating behaviour remains unqualified
on GPU, which matters directly for the longer-window run in claim 8.

**Correction 7b — separate capability from use.** Unused in this run:
`register_hypothesis` (0 invocations; CD82 instead wrote a `hypotheses.md` note),
the contingent controller's no-model-call continuation (0 contingent plans, all
5 batches length-1, `eligible_detector_attempts: 0` in all three games), and vision.
Notes were written in 2 of 3 games (LS20 and CD82 `NOTE_WRITE`; S5I5's
`integrity.jsonl` has only `GUARD_INSTALLED`). So "notes/helpers/hypothesis
registration" and "contingent controller can execute one matching continuation
without a model call" are accurate as *available capabilities* and should be
labelled as such — this run demonstrates neither.

### 8. Proposed tests — **adequate in direction, under-specified in configuration**

The proposed public run (10 min admission/game, same 3 games, cap 10→200, same
model/solver, sequential, reserves kept, 1 h global cap, first-vs-second 5 min
comparison) targets the right unknown: whether the marginal action rate after
orientation survives a long window, and what history growth does to latency. I
support running it. Four concrete problems:

**(a) The outer guards would exceed a 1-hour session.** `build.py:88` sets the
notebook's `subprocess.run(timeout=…)` to **6300 s** on the public path, and
`exact_request_outer_runtime.py:2194` guards the probe with `launcher_timeout_s`.
If the session cap is 3600 s and `--launcher-timeout-s` is left at 4800 (or raised
to 3600), a hang would be terminated by Kaggle, not by an orderly internal timeout
— destroying exactly the clean-shutdown evidence this whole line of work exists to
produce. Both guards must be brought *inside* the session cap.

**(b) Exact changes required** (source-derived; not implemented):

| # | File:line | Now | Needed |
| --- | --- | --- | --- |
| 1 | `submission-20260910-v5/build.py:72` | `SUBMISSION_ADMISSION_CAP_SECONDS = '75'` | `'600'` |
| 2 | `build.py:50-58` env dict | *(unset)* | add `'SUBMISSION_ACTION_LIMIT': '200'` — `play.py:76` otherwise defaults to `'10'` on the public path |
| 3 | `build.py:85` | `… else '4800'` | ≥ **2445** for the 600 s cap to bind (`admission_budget(T−60, 3, 90) ≥ 600 ⟺ T ≥ 2445`), and ≤ session cap − startup; **≈2700** |
| 4 | `build.py:88` | `timeout=… else 6300` | ≈**3100** (inside a 3600 s session) |
| 5 | new card | — | the frozen card's Cap clause names 75 s / 10 actions; write a new card, do not edit the old one |

Nothing else is needed, and I checked the traps: `build.py:19` `max_actions='10'` is
overwritten by `play.py:79`; `worker_timeout_s=300` is overwritten per game by
`play.py:100-102` (which would become 600+90+15 = 705 s); and
`nooa_public_probe.py:54 PUBLIC_MAX_ACTIONS` is only *reported* (`:507`), never
enforced, so 200 is safe for accounting. `headless.py:53 --max-actions-per-turn`
stays 20 — a per-turn batch cap, unrelated to the 200-action game cap.
`adapter.py:114` `MAX_ACTIONS = 2·limit + 1` moves 21 → 401 as intended.

**(c) It does not change only the intended limits.** `solver_agent.py:1470-1473`
injects `model_admission_seconds_remaining=…` and "Budget stops further model
calls…" into the game context on every turn. At 600 s the model sees a different
prompt than at 75 s. This is a different treatment, not a longer instance of the
same one, and the first/second-5-minute comparison inherits that. Say so in the
card rather than reading the result as a pure extension of the qualified run.

**(d) A competition-sized *per-game* budget on 3 games does not qualify the
110-game allocator.** With `game_count=3` the schedule is nothing like the 110-game
schedule (table in claim 6). The rolling allocator can only be qualified either by
the competition rerun itself or offline against 110 supplied environments with a
stub agent — `simulate_allocator.py` covers only the arithmetic.

**Cheaper, more discriminating alternatives — do these first, they cost no GPU:**

1. **Failure containment locally, before any GPU run.** The proposed injected
   slow-HTTP / error / stalled-tool tests need no GPU and no paid model; the
   existing harness already drives the real Agent loop with controlled model
   responses (`test_submission.py::test_real_entrypoint`). Buying containment
   evidence with GPU hours would be a waste.
2. **Characterize prompt growth locally.** Run the controlled-response fixture out
   to 50–100 turns and record `prompt_tokens` per call. This settles what
   `TruncationConfig(max_context_tokens=20_000)` actually bounds — the single
   biggest source of uncertainty in claim 6 — for free, and it tells you whether
   the 600 s window will even be spendable before you spend it.
3. **Turn the GPU run into a dose-response, not three replicates.** Give the three
   games **300 / 600 / 1200 s** admission instead of 600 each. Same session, same
   cost (35 min of admission vs 30), and you get both the within-game first-vs-second
   window comparison *and* a direct window-length endpoint comparison. Three
   replicates at one dose mostly re-measure the orientation phase three times.
4. **Pre-register the measures**, since with 3 games nothing here is statistically
   powered: per-call `prompt_tokens` trajectory and whether it plateaus; calls
   before vs after the first action; `wall_seconds` vs `prompt_tokens` regression;
   `step_checks` status distribution (does any prediction ever get committed —
   correction 7a); contingent-plan count; peak worker RSS against the 4 GiB
   `RLIMIT_AS` in `headless.py:111-117`; and the same lifecycle reconciliation.

**Recommendation:** do (1) and (2) now at zero GPU cost; then one GPU run shaped as
(3) with guards fixed per (b) and a new card; do not read any of it as licensing a
score forecast.

---

## Findings, severity-ranked

| # | Sev | Finding | Evidence |
| --- | --- | --- | --- |
| F1 | High | Claim 7's "predictions checked / batches gated on predictive agreement" describes a mechanism that returned `prediction not committed` → `unresolved` on **all 5 batches in all 3 games**. Behaviourally harmless here (all batches length-1, `skipped_actions: 0`) but the description is wrong about this run, and the gate stays unqualified on GPU. | `solver-workspaces/*/ipc/step_checks.jsonl`, `batch_stops.jsonl` |
| F2 | High | Claim 6's "228.545 s/game" reads as uniform. The allocator is harmonically back-loaded: **91.909 s** admission for game 1 vs **710–785 s** for game 110, because 135 s/game of unspent reserves recycle into the pool. The *mean* (~225.8 s) and the total are fine. | `play.py:26-40,84-102`; `simulate_allocator.py` |
| F3 | Med | Claim 7 mixes capability with use. Unused in this run: `register_hypothesis`, contingent continuation, vision; notes in 2 of 3 games. Label them as available capabilities. | tool payloads (50× `execute_python`), `quick_metrics`, `integrity.jsonl` |
| F4 | Med | Proposed run's outer guards (`build.py:88` = 6300 s; probe guard = `launcher_timeout_s`) sit outside a 1-hour session cap, so a hang ends by Kaggle kill, not orderly timeout. Plus `SUBMISSION_ACTION_LIMIT` is currently unset, so the 10→200 change needs a new env entry, not just a cap edit. | `build.py:85,88`; `play.py:76`; `exact_request_outer_runtime.py:2194` |
| F5 | Low-Med | 371.30826 s is a mixed-clock difference (notebook-relative minus launcher-monotonic) and omits the unlogged head/tail. Report as **≈371.3 s** with the four-part decomposition; `329.128 s` is the clean single-clock figure. | `events.log`, `result.json`, terminal log |
| F6 | Low | "Version 4" is not derivable from preserved evidence — no artifact carries a kernel version ordinal, and the remote readback is source-only. Source identity *is* established by hash. | `remote-source/`, `raw/`, `STATUS.md` |
| F7 | Low | `audit.py:57` scans an explicit allow-list of failure patterns, not "any error". A wider scan finds 6× `Failed to get device capability: SM 12.x requires CUDA >= 12.9` at server startup — benign (all 35 requests returned 200) but outside the audit's field of view. Say "no marker from the enumerated patterns", not "no error". | `server.log:1-10` |
| F8 | Info | Reserves are enormous relative to observation: total tool wall time is **0.032 / 0.040 / 0.043 s per game** against a 15 s tool margin, and non-model per-game time is ~4.2 s against a 30 s cleanup reserve. This is the direct cause of F2's back-loading. Not a defect — it is what makes the tail safe — but the tuning knob is identified. | `tool_calls.jsonl`, `model_calls.jsonl` |
| F9 | Info | p90 = 13.729653 is linear-interpolation; nearest-rank 13.811175, exclusive 13.877570. n=34 with one 28.6 s outlier. | `recheck.py` |
| F10 | Info | Model-**file hashing** (175.6 s) exceeds model **loading** (148.1 s) in the startup budget — the largest single reducible item, and one a long run amortizes. | `events.log` |

## Remaining uncertainty

- **Kaggle-side facts** (version ordinal, submission state, quota display) cannot be
  checked from this machine under the authorized scope. They stay coordinator
  assertions.
- **History growth over a long window** is the load-bearing unknown behind every
  110-game number. Unresolved; addressable locally at zero cost (recommendation 2).
- **Whether the prediction gate ever resolves** on GPU — F1 means it has never been
  observed doing so in a public run.
- **The 110-game allocator schedule** is verified only arithmetically here. My
  simulation assumes each game consumes its full admission window plus a constant
  overhead; a game that wins or stalls early returns more time and shifts the
  schedule further toward the tail.

---

## Addendum — corrections to this review (2026-09-10)

Raised by the coordinator after delivery. I checked each against the evidence
before responding. **All five are correct; I concede all five.** This addendum
supersedes the passages it names. `RESULT.md` was not touched.

### A1 — Recommendation 2 (local prompt-growth characterization) is unsound as written. Conceded.

`deployment/kaggle/submission-20260910-v5/test_submission.py:111` hardcodes
`usage=dict(prompt_tokens=22000, completion_tokens=50, total_tokens=22050)` in the
fixture's fabricated response. Every `model_calls.jsonl` row from that harness would
record 22000 regardless of the real message list, so running it out to 50–100 turns
measures nothing about prompt growth. Worse, the fixture's replies are fixed-length
canned tool calls, so even the underlying history would not grow the way live play
grows it.

What survives is much narrower, and I should have said only this:

- **Free and sound:** determine by *source inspection* what
  `TruncationConfig(max_context_tokens=20_000)` (`headless.py:126-130`) actually
  clamps, given that observed prompts reached 29,906. That is a mechanism question,
  answerable by reading nooa's truncation implementation, with no run at all.
- **Sound but bounded:** serialize the real request payloads and count them with the
  pinned tokenizer rather than trusting fixture `usage`. That fixes the measurement
  instrument; it still cannot produce the trajectory, because each turn's content
  depends on what the live model wrote in the previous turn.
- **Not obtainable locally at all:** the live growth curve. My claim that this
  "settles the single biggest source of uncertainty for free" was wrong. It is
  settled only by the GPU run.

### A2 — Recommendation 3 (300/600/1200 s across the three games) is confounded. Withdrawn.

LS20, S5I5 and CD82 are different games, so assigning a different window to each
confounds duration with game identity at n=1 per cell. It is not a dose-response and
I should not have called it causal. Two further errors in that paragraph: "same cost
(35 min of admission vs 30)" is self-contradictory — 35 > 30, it is *more* cost; and
"strictly more information" is unsupported.

**The coordinator's original design is better identified on this axis than my
alternative.** Comparing the first and second 5-minute windows *within* each game is
a within-unit contrast that holds game identity fixed, which is exactly what the
confounded between-game design gives up. Recommendation 3 is withdrawn; run all
three at the same window. A genuine window-length contrast requires repeating the
same game at two durations, which costs additional session time and was not what I
priced.

### A3 — "model loading is 148.1 s" is wrong, and the correct number is far more favourable to claim 4. Conceded.

148.126 s is SGLang **launch + warmup**, not isolated model loading. The raw records
decompose it exactly:

| Component | Source | Seconds |
| --- | --- | ---: |
| launch → API ready | `model-api-readiness.json` `elapsed_s` | 140.026 |
| warmup chat completion + `freeze_gc` | `warmup-facts.json` `elapsed_s` | 8.099 |
| **total** | | **148.125** (= logged 148.126) |

And inside the 140.026 s, from `server.log`:

| Step | Seconds |
| --- | ---: |
| process spawn → first server log line (23:46:04.76 → 23:46:32) | ~27.2 |
| Init torch distributed | 8.04 |
| **Load weight** (`type=Qwen3_5ForConditionalGeneration, quant=fp8`) | **2.57** |
| Capture target prefill CUDA graph | 83.11 |
| Capture target decode CUDA graph | 3.65 |
| kv cache allocation | 0.53 |

So **actual model weight loading is 2.57 s** — 1.7% of the 148.1 s and 0.7% of the
371.3 s residual. The dominant item is CUDA graph capture (86.76 s, 58% of startup).
This strengthens the coordinator's claim-4 assertion well beyond my phrasing: the
residual is not merely "not all model loading", it is **almost none** model loading.
Read alongside F10, the startup budget is graph capture (86.8 s) and file hashing
(175.6 s), with weight load a rounding error.

### A4 — F2's back-loading is conditional simulation, not a guaranteed property. Conceded, and narrowed.

`simulate_allocator.py` assumes every one of the 110 games consumes its full
admission window plus a constant overhead. That assumption held for all three
observed games, but at 110 games it is an assumption, not an observation: a game
that wins, stalls, or errors early returns time and shifts the schedule further
toward the tail, while any path that overruns shifts it the other way. F2 should
read *"under full-window consumption, the arithmetic implies a 91.9 s → ~750 s
spread"*, not *"the allocator is harmonically back-loaded"*.

I also accept that the source table said **average gameplay time**. I could not
inspect that table — `REQUEST.md` claim 6 paraphrases it as "25140 gameplay seconds
-> 228.545 s/game" without the qualifier, and I reviewed the paraphrase. Taking the
coordinator's statement at face value, my framing objection does not apply to the
original text, and F2's severity drops accordingly. **What survives F2 is one
distinction only:** the per-game *wall* share is not the per-game *model-admission
window* — at 110 games the first window is 91.9 s of a 226.9 s share, the 135 s
difference being reserves. That distinction is worth keeping; the "misleading
framing" charge is withdrawn.

### A5 — F1/F3 overreached on scope. Conceded.

- My scope is this run. Statements that the prediction gate "stays unqualified on
  GPU" and that its "discriminating behaviour remains unqualified" generalize across
  runs I did not examine. Withdrawn. The supported statement is exactly:
  **in this run, all five `step_checks.jsonl` rows across the three games returned
  `unresolved` / `ValueError: current prediction is not committed`, so no observation
  was compared against a committed prediction, and no batch was truncated.** F1's
  correction to the wording stands on that basis alone.
- The contingent item in claim 7 was already framed modally — "contingent controller
  **can** execute one matching continuation without a model call" — so F3 over-fires
  there and I withdraw it for that item. F3 stands for the non-modal phrasings:
  "observations/predictions checked; batches gated on predictive agreement" and
  "notes/helpers/hypothesis registration", where this run shows `register_hypothesis`
  never invoked and notes written in 2 of 3 games. The coordinator's point that the
  unresolved outcomes are a valuable *additional* qualification rather than a
  contradiction of a capability claim is the right reading, and I adopt it.

### A6 — Arithmetic magnitude error in the claim-6 rate-convention note. Conceded.

I wrote that the two rate denominators differ by "0.01 action/game". Recomputed:

| | launcher clock (5/255.019576933) | game-clock sum (5/255.012660413) | difference |
| --- | ---: | ---: | ---: |
| total actions | 492.9033352 | 492.9167038 | **0.0133686** |
| per game | 4.4809394 | 4.4810609 | **0.0001215** |

So ~0.013 actions in total and ~0.00012 per game — roughly 100× smaller than I
stated. The choice of denominator is numerically irrelevant; only naming which one
was used still matters, and that is a documentation point, not a correction.

### Net effect on the review

Nothing here changes the verdicts on claims 1–5. Claim 4's attribution is
**more** strongly supported than I stated (A3). Claim 6's status changes from
"verified as arithmetic; one framing is materially misleading" to **"verified as
arithmetic; one distinction worth adding"** (A4, A6). Claim 7's correction 7a stands
on this run's records with its scope narrowed (A5); 7b narrows to the non-modal
items. Claim 8's assessment stands except that my recommendation 2 is narrowed (A1)
and recommendation 3 is withdrawn in favour of the coordinator's within-game
comparison (A2) — leaving the substantive claim-8 findings unchanged: the outer
guards (`build.py:88` 6300 s; probe guard = `launcher_timeout_s`) sit outside a
1-hour session cap, `SUBMISSION_ACTION_LIMIT` is unset so 10→200 needs a new env
entry, the prompt text the model sees changes with the window, and 3 games cannot
qualify the 110-game allocator.
