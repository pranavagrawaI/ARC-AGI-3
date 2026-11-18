# T5 launch and submission result — actor-critic competition v1 (2026-09-17)

Status: **push complete, competition submission accepted, rerun terminal = ERROR** (read back 2026-09-17T12:15:58Z and 12:16:16Z). No score. No retry made or authorized. This file is written only from read-back receipts.

## 1. Authorization and preflight

| Item | Value / receipt |
|---|---|
| Human authority | user request for one competition submission with the actor-critic solver; construed by T5 as one push + one submission (`AUTHORITY-NOTE.md`) |
| Live preflight | `LIVE-PREFLIGHT.json`, 2026-09-17T11:16:02Z, verdict GO, 0 failures |
| Bindings recomputed | manifest `dcb716bc…`, notebook `3f38ad9b…`, card `eb8ea216…`, review `b18ee233…` (APPROVED, independent), CHASSIS contract `9431c419…` — all exact |
| Gate reconciliation | OK over 85 manifested sources; gate `check` before authorization = HOLD on missing `authorization.json` only |
| Provider state before | authenticated as `pranavagrawai`; `numAllowedNow=1`; `No submissions found`; GPU quota 4.08 h used / 25.92 h remaining of 30 h (refresh 2026-09-19T00:00Z); competition entered |
| Target absence | `GetKernel` on own slug → 403 (calibrated: own existing private kernel resolves); `kernels list --mine --search <slug>` → Not found |
| Metadata | private, GPU, `NvidiaRtxPro6000`, internet off, TPU off, sole competition source `arc-prize-2026-arc-agi-3`, exact four datasets, image pinned by digest, title slugifies to id via installed SDK |
| Tooling | Kaggle CLI 2.2.4 / kagglesdk 0.1.37; `--timeout` and `--accelerator` supported |
| Authorization | `authorization.json` (sha `91ed5546…`), exact gate schema only; `SUBMISSION-AUTHORIZATION.json` separate, bound to version 1 only after readback |
| Gate `check` | READY at 11:16:50Z (`launch/gate-check.txt`); pre-push recheck of hashes/limits/absence at 11:16:50Z (`launch/prepush-recheck.txt`) |
| Waiver | no exact-byte remote public-mode run exists for this package; accepted as a limitation, not evidence (`AUTHORITY-NOTE.md`) |

## 2. Push / commit identity and terminal result

| Item | Value |
|---|---|
| Invocation | `launch_control.py push` only; 11:17:02Z → 11:17:06Z, exit 0 (`launch/push-invocation.txt`) |
| Consumption | `CONSUMED.json` written O_EXCL before the provider call; gate now returns `HOLD: Consumed authorization` — a second push is mechanically impossible |
| Provider receipt | `push-receipt.json`: "Kernel version 1 successfully pushed" → `https://www.kaggle.com/code/pranavagrawai/arc3-ac-competition-20260917-v1` |
| Server readback | `launch/server-readback-v1-post-push.json` and `-pre-submit.json`: PASS. ref/slug/title exact, kernel id 134721414, `current_version_number=1`, private, GPU, `NvidiaRtxPro6000`, internet off, TPU off, competition source exact, four datasets exact, image digest exact |
| Source equality | parsed notebook == frozen notebook; executable code sha `7319ee0c…` identical both sides; raw bytes differ only by CLI reserialization (server 872,723 B vs frozen 872,825 B) |
| Commit run | QUEUED 11:17→RUNNING 11:23:09Z→COMPLETE 11:24:11Z, failure_message empty (`launch/monitor-commit-v1.jsonl`) |
| Status scope | exact `version_label=1` session-status returns 404 both while queued and after completion (same as R9 receipts); status bound through current identity with `current_version_number==1` confirmed each tick and no second version ever existing |
| Commit output | exact `version_number=1` download HTTP 200: `submission.parquet`, 2,648 B, sha `71bfd543…`, 1 row `[1_0, 1, True, 1]`, columns row_id/game_id/end_of_game/score, dtypes str/str/bool/int64; output listing shows only that file (`launch/commit-output-v1.json`, `launch/retrieved-v1/`) |

## 3. Competition submission readback

| Item | Value |
|---|---|
| Pre-submit recheck 11:24:50Z | `numAllowedNow=1`; `No submissions found`; readback PASS again; all local hashes exact (`launch/presubmit-recheck.txt`) |
| Command (once) | `kaggle competitions submit arc-prize-2026-arc-agi-3 -k pranavagrawai/arc3-ac-competition-20260917-v1 -v 1 -f submission.parquet -m "Actor-critic solver v1"`; 11:25:02.9Z → 11:25:07.6Z, exit 0, stdout `0 submissions remaining today.` (`submission-receipt.json`, `SUBMISSION-CONSUMED.json`) |
| Readback row | ref **56303234**, file `submission.parquet`, description `Actor-critic solver v1`, date 2026-09-17T11:25:04.373Z (inside the command window), status PENDING, submitted_by `pranavagrawai`, url `/code/pranavagrawai/arc3-ac-competition-20260917-v1?scriptVersionId=350552552` (`launch/submission-readback-1.txt`, `launch/submission-readback-sdk-1.json`) |
| Limits after | `numToday=1, numTotal=1` |
| Monitor | detached read-only `launch/tools/monitor_submission.py 56303234 36000 300` → `launch/monitor-submission-56303234.jsonl` (5-min polls); PENDING through 12:10Z, **ERROR at 12:15:58Z** |
| Terminal readback (SDK, 12:16:16Z) | ref 56303234 status `ERROR`, `error_description`: "A system error. Please try resubmitting to resolve the error and contact Kaggle Support if it persists."; public/private score empty; kernel `current_version_number` still 1; kernel `last_run_time` time-of-day 12:16:14 (SDK date field is offset; time matches the rerun end) (`launch/submission-readback-sdk-terminal.json`) |
| Exposed outputs after ERROR | current-identity output listing unchanged: `submission.parquet` only; the 16-line stderr log is the commit run's nbconvert boilerplate (`launch/commit-v1-log.json`); **the rerun log and the accelerator actually allocated are not exposed by the API** (`launch/post-error-output-listing.json`) |
| Daily slot after ERROR | `numAllowedNow=1` again (`launch/submission-limits-after-error.json`): Kaggle refunds an errored submission. Recorded as fact; it does not extend this task's authority. |

## 4. Local evidence vs remote observation vs scientific outcome

- **Local evidence (exact):** bytes frozen and unchanged through launch (manifest reconciliation, drift `[]`, unmanifested `[]`); independent review APPROVED on these exact bytes; local fixture qualification per `README.md`/`REPAIR-A1.md`.
- **Remote observation (exact):** provider holds the same executable bytes under the exact identity; the non-rerun commit completed and produced a schema-valid dummy parquet; the competition accepted one submission bound to that version; the rerun ended ERROR ~50–51 min after submission with Kaggle's generic "system error" text and no score.
- **Timing inference (bounded):** submission 11:25:04Z; still PENDING at the 12:10Z poll; ERROR by 12:15:58Z; kernel last-run time-of-day 12:16:14. Elapsed ≈2,750–3,050 s including queue. The package's startup envelope is `BOOTSTRAP=1800` + `STARTUP_WITH_WARMUP=1200` = 3,000 s (`budget.py`; `entry.py` runs `install.py` with `timeout=1800, check=True`, then the launcher with `--startup-timeout-s 1200`). R9 on the same requested shape spent ≈780 s in bootstrap+startup (2,303 s total wall, 1,522 s model-runtime phase). So the rerun died inside startup and never reached gameplay deadlines (gameplay budget 28,020 s). Consistent hypotheses, not discriminated: (a) model startup timed out at 1,200 s after a normal bootstrap (e.g. accelerator substitution to a 16 GB P100 as on 2026-09-03 — the 28.7 GB checkpoint cannot load); (b) bootstrap hit its 1,800 s timeout; (c) a Kaggle-side rerun fault matching the boilerplate text literally. The discriminating evidence (rerun log, `nvidia-smi` in the rerun, allocated shape) is not in the API.
- **Scientific outcome:** none. The card's feasibility question ("does the package complete the rerun and yield submission.parquet within 9 h?") is answered **no for this attempt**, at the startup stage, with the cause unresolved. Card result map: failed valid run → preserve evidence and diagnose without retry.

## 5. Spend / quota

| When | GPU used | GPU remaining |
|---|---|---|
| Preflight 11:16Z | 4.08 h | 25.92 h |
| Pre-submit 11:24:50Z | 4.08 h | 25.92 h (commit run used no GPU time visible at this granularity) |
| After rerun ERROR 12:16Z | 4.08 h | 25.92 h — the competition rerun is **not** billed to account GPU quota (`launch/quota-after-submission-56303234.json`) |

Whether the competition rerun is billed to account GPU quota is not established; the card's 9 GPU-h cap is provider-side (`--timeout 32400`).

## 6. No-retry statement

Exactly one gated push (version 1) and exactly one competition submission (ref 56303234) were made. `CONSUMED.json` and `SUBMISSION-CONSUMED.json` exist; `authorization.json.used` stays `false` as the gate requires while `CONSUMED.json` blocks reuse; `SUBMISSION-AUTHORIZATION.json.used=true`. The rerun ended ERROR. Kaggle's error text invites resubmission and the daily slot was refunded; neither extends this task's authority. No repush, resubmission, alternate slug or version was made. Any retry is a new user decision and should follow a diagnosis, not precede one.

## 7. First unresolved boundary

**Which startup stage failed in the rerun, and on what accelerator.** The API exposes no rerun log and no allocated shape. Two cheap ways to resolve it, both outside this task's authority: (1) open the submission's log in the Kaggle UI (`/competitions/arc-prize-2026-arc-agi-3/submissions`, ref 56303234) and read the last lines of stderr / the `nvidia-smi` block that `install.py` and the launcher print; (2) run the exact bytes once in public (non-rerun) mode with a forced `KAGGLE_IS_COMPETITION_RERUN`-equivalent fixture — which this package deliberately does not offer (waiver in `AUTHORITY-NOTE.md`), so it would be a new candidate with a fixture flag, not this one. Until one of these is read, the accelerator-substitution hypothesis and the bootstrap-timeout hypothesis are both live, and the review residual "no remote public-mode path" has become the operative gap.

## Receipts

`launch/RECEIPT-INVENTORY.json` holds SHA-256 and byte counts for every authority, launch, readback, monitor and submission receipt as of its `utc`. Scripts used are preserved under `launch/tools/`. No candidate, card, manifest, review, test or launch-tool bytes were modified. No credentials, auth headers or signed URLs are recorded; the submission `url` field is a relative Kaggle path.
