# q6 speed probe — 20260905

One question: **what is the q6 candidate's real per-decision wall time on Kaggle FP8?**

No such measurement exists. q6 was qualified with `"platform": "scripted"`, `"model_calls": 0`,
`"model_wall_seconds": 0` — a stub, never a real model. The q6-control-preflight's 200-decision cap
was set without it, and the only real-model reading on the shared route
(`prototype-nooa-upstream-public3`, see `../../analysis/deployment-probes/20260905-q6-control-preflight/THROUGHPUT-EVIDENCE.md`)
was 5 solver decisions in 2h45m, one of them on ls20.

## Delta from the frozen control

Built by the qualified builder from the same contract. Four changes, nothing else:

| Knob | Control | Probe |
| --- | --- | --- |
| heading | q6 candidate Phase A | speed probe |
| `TYCHO_KAGGLE_MAX_ACTIONS` | 200 | **10** |
| `TYCHO_NOOA_WORKER_TIMEOUT_SECONDS` | 2400 | **900** |
| `--launcher-timeout-s` | 7200 | **1800** |

`ARC_STALL_REVIEW='0'` (control), `ls20` only, `--request-timeout-s 180`, thinking disabled,
4096 output tokens, offline, no submission — all unchanged.

## Why it is informative either way

- 10 actions complete quickly -> a direct decisions-per-hour figure for the candidate.
- It stalls -> the 900 s worker timeout fires having recorded N decisions and their timings, giving
  a latency lower bound and a second instance of the stall itself.

Neither branch is silent, which is the point.

## Bounds

Private kernel, one push, no automatic retry, `NvidiaRtxPro6000`, 3600 s session ceiling.
Worst case is roughly 280 s setup + 150 s warmup + 1800 s gameplay, under one GPU-hour.
Quota at push: 6.47 h used, 23.53 h remaining.

## What it cannot establish

Not a q6 result. Establishes no stall incidence, no treatment efficacy, no level progress, and
does not count as a replicate anywhere. It measures latency and nothing else. The
q6-control-preflight identity and its single push remain unspent.

## Record

Pushed 2026-09-05T15:49:08+05:30, `pranavagrawai/q6-speed-probe-20260905`, version 1 accepted.
Offline request check: `verify_probe.py`, PASS, 0 transport calls.
