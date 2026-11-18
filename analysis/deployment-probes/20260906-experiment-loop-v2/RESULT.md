# Experimentation iteration result

Executed 2026-09-05; the directory's 20260906 label is not the execution date. The isolated workflow improved completion of measured experiment loops in these short starts. It did not establish reliable causal explanations or improved game-solving. Six live starts, 72 calls, provider Reka serving qwen/qwen3.8-27b, zero reported reasoning tokens, total reported cost **$0.278880906**. All starts ended at the precommitted 12-call cap. No Kaggle launch or queued-package mutation occurred.

| Condition | First accepted action, call | Completed regional loops | Actions |
|---|---:|---:|---:|
| Control A, current-state fix | 8 | 0 | 1 |
| Control B, current-state fix | 4 | 1 | 1 |
| Workflow v1 A | 2 | 2 | 2 |
| Workflow v1 B | 2 | 2 | 3 |
| Workflow v2 A | 2 | 2 | 2 |
| Workflow v2 B | 5 | 2 | 3 |

Each start had the same local ls20 initial state and deliberately provisional wrong hypothesis: UP changes the top-right corner. Control and v1 ran in balanced order control-a,workflow-a,workflow-b,control-b. V2 was a subsequent exploratory revision, not a randomized comparison. Controls were manually judged from their raw traces; they were not assigned zero merely because they lack the intervention's log. See ../20260906-experiment-loop-v1/CONTROL-ADJUDICATION.json.

A regional loop means a pre-action change/no-change prediction with an opposing alternative, one accepted action, inspection of its actual successor, a correct supported/rejected conclusion, and a next hypothesis/test. This is narrower than proving an object identity, direction, inverse relationship or causal mechanism. Workflow logs are independently recomputed from raw frames; recorded verdicts are not trusted.

## Changes retained as an isolated candidate

- **Current-state delivery:** state notifications wake the actor, but their embedded snapshots are replaced with the latest host state. Repeated stale-only wakeups are suppressed; user messages remain deliverable. Both initial comparison arms received this correction.
- **Persistent experiment record:** experiment() commits one specific regional prediction and opposing alternative before its action. Raw action submissions are rejected in this experimental workflow. A new experiment is blocked until the pending result is closed.
- **Host-measured comparison:** experiment_result() uses the recorded action's pre/post frames. The actor must acknowledge contradiction and change its hypothesis. Honest rejection counts as informative progress.
- **Complete change summaries:** v2 reports exact connected-component counts and inclusive bounds for all changed pixels, replacing the truncated sample that the model had mistaken for the whole difference. It explicitly limits what the regional predicate establishes.
- **Usable history:** trajectory(per_step=True) now reads worker transitions and includes both before_grid_rows and grid_rows. Animation frames are explicitly unavailable, not invented. The inherited method read a native logger file absent on this host and returned an empty list.
- **Interface/cleanup corrections:** v2 makes the supported/rejected argument a Literal with an explicit example, and shuts down queue producers on headless exit.

The source is experiment_agent.py, state_delivery.py and headless.py in this directory. These import the unchanged frozen q6 chassis. SOURCE-PIN.json covers source and each run's precommitted card. The source/card hashes remained unchanged during paid execution.

## What the evidence supports

All four workflow starts completed two correctly classified regional loops; the controls completed zero and one. V1 started acting earlier in both observed starts. That is enough to retain the mechanical workflow as promising. V2 preserves the closure count and removes observed conclusion-format repair detours. It did not consistently improve first-action latency: one start waited until call 5 despite the instruction to act within three calls.

The model correctly rejected its initial wrong regional prediction. With v2, UP was reported as 50 changed pixels in rows 40–49,columns 34–38 plus 2 in rows 61–62,column 13. The model subsequently tested DOWN and, in one start, correctly supported the narrower claim that the same main region changes.

## What does not follow

The stronger grounded-explanation criterion remains unmet. V1 invented linked-display explanations and treated a sample as the full change set. V2 sometimes still invented object positions unrelated to the measured components, called a 10x5 region 8x5, or inferred reversal/identity from mere regional change. A correct close_experiment call can be produced by copying the supplied required_conclusion; it is evidence of executing the measurement workflow, not independent scientific judgment. We are not promoting those closures to a claim of causal understanding.

The loop interfaces enforce sequence and a narrow observable claim. They do not verify whether two narrative hypotheses are exhaustive or whether the chosen test identifies their cause. No level was completed. The model/provider route and local environment hashes differ from the exact Kaggle FP8 execution. These traces are not q6 research-arm replicates.

## Local qualification and informative failures

V1: 3 tests passed, including real Headless/CodeAct with fake replies completing two loops after a contradicted prediction. V2: 4 tests passed in 2.29 s, adding exact change-component and full-history checks. Ruff passed on the new Python files. The only final warning was the inherited set_dynamic deprecation.

An early v2 test reached both closures but hung during asyncio shutdown because the actor's queue producer outlived the headless task. The preserved INITIAL-TEST-SHUTDOWN.txt and temporary trace showed completed loops before shutdown. Explicit queue_manager.shutdown() fixed cleanup; the full v2 test then passed. No paid run used the failing cleanup version.

Independent scorer checks on copied live artifacts passed: changing stored verdict fields did not change the score; changing a closed prediction relative to its committed record was rejected. SCORER-CHECK.json records this. Both versions' source/card pins and the queued package's prepare.py --check passed after execution.

```bash
deployment/kaggle/.venv/bin/python -m pytest analysis/deployment-probes/20260906-experiment-loop-v2/test_loop.py -q
python3 analysis/deployment-probes/20260906-experiment-loop-v1/score.py
python3 analysis/deployment-probes/20260906-experiment-loop-v2/score.py
python3 deployment/kaggle/q6-control-preflight/prepare.py --check
```

## Decision and resume

Retain v2's reliable state/history/measurement machinery as the isolated incumbent. Do not replace the queued Kaggle package, and do not claim that v2's richer prompt outperforms v1 or that causal reasoning improved. The next decision-changing trial should test whether a proposed second experiment discriminates a specific causal alternative—such as reversal versus another change—using an exact measured relation, rather than asking for more prose. It needs a new card; these trials are complete and will not be extended.

COMPARISON.json contains all six counts and costs. SCORE.json contains independently checked workflow closures. requests.jsonl, responses.jsonl and each run's workspaces preserve raw evidence. READABLE-TRACE.txt deduplicates feedback and expands literal escape sequences solely for human reading; the JSONL remains authoritative. No secrets were included in request logs, no paid retries were made, and no live probe remains running.
