# q6 control preflight — launch refused at admission, 2026-09-05T11:27:22+05:30

Not a run, not a failure of the candidate, and not a consumed push. Kaggle rejected the
request before creating a kernel version.

```
Kernel push error: Maximum batch GPU session count of 2 reached.
```

## Preflight gates all passed

| Gate | Result |
| --- | --- |
| `prepare.py --check` | exit 0; binding identical to `LAUNCH.json` |
| `verify_request.py` | PASS; 0 transport calls; private, GPU, no internet, 14400 s, `NvidiaRtxPro6000` |
| `kaggle quota` | 30.00 h remaining of 30.00 h, refresh 2026-09-12 |
| target slug pre-push | inaccessible — never pushed, one-push cap intact |

## State after the refusal is clean

- `kaggle kernels status pranavagrawai/q6-control-preflight-20260905` -> **404**, no version exists.
- `kaggle quota` -> GPU used **0.00 h**, unchanged.

Nothing was created and nothing was consumed. `maximum_pushes: 1` is still unspent. This is an
unambiguous server-side refusal, not an ambiguous network response, so the no-retry rule is not
in question — but the blocker is capacity, so re-pushing now would be refused identically.

## The two batch slots are held by queued reference arms

| Kernel | Status | Last run | Shape |
| --- | --- | --- | --- |
| `pranavagrawai/prototype-tycho-upstream-public3` | QUEUED | 2026-09-05 04:07:53 | `NvidiaRtxPro6000` |
| `pranavagrawai/prototype-nooa-upstream-public3` | QUEUED | 2026-09-05 04:07:45 | `NvidiaRtxPro6000` |

Both have been queued roughly 7.3 hours without allocation, consuming zero quota, and both
request the same accelerator the preflight requires. Kaggle counts a QUEUED batch session against
the concurrency limit of 2.

The Kaggle CLI has no cancel or stop subcommand. `kernels delete` removes the notebook itself and
is not a session control. Releasing a slot requires the Kaggle web UI.

## Bearing of the 2026-09-03 record

`analysis/deployment-probes/20260903T-vision-readback-v16/TERMINAL-RESULT-kernel3.md` records the
same accelerator request being *accepted* and allocated a `Tesla P100-PCIE-16GB`, failing
accelerator preflight in 0.041 s. Today's arms are queued rather than mis-allocated, which is a
different mode; neither establishes that a genuine RTX PRO 6000 is obtainable now. Freeing a slot
buys an admission attempt, not an allocation.

## Gap this exposes in the preflight

The package verified request bytes, CLI construction and quota. It did not check batch-session
admission capacity, which is what actually refused the launch. Quota and concurrency are separate
limits, and a queued session shows 0.00 h against quota while still holding a slot — so the
card's quota read cannot detect this condition. A concurrency check belongs in the go/no-go list
before any future push.

## Disposition

Blocked on batch GPU session concurrency. Bytes frozen and verified; the launch binding is
unchanged and remains valid for a single push when a slot is free. No automatic retry, no
schedule change, no accelerator substitution — the card makes a wrong accelerator an invalidating
condition.

## Decision, 2026-09-05

Hold and monitor. Both reference arms keep their queue positions; no arm cancelled, no accelerator
substituted, no re-push attempted. A watch polls both slugs every 5 minutes and reports any status
transition, exiting when either reaches COMPLETE, ERROR or CANCEL — a slot frees only on a terminal
state, since a RUNNING batch session still holds one. Launch requires fresh authorization plus a
re-run of the hash and quota checks at that time.
