> Diagnosis update: the official launcher returned 0. Exit 74 was assigned by `nooa_public_probe.py` after a stale hardcoded chassis identity failed post-run validation. It was not interpreter finalization. See [exit diagnosis and tested repair](exit-review/RESULT.md). The original observations below are retained.

# q6-summary-boundary-20260906 — the repair worked live; the run died on its exit code

`pranavagrawai/q6-summary-boundary-20260906`, pushed 13:29:11Z, no queue, terminal **ERROR**
14:26:42Z. Probe `elapsed_s` 3369.95 against a 10800 s session cap. Genuine RTX PRO 6000.

**The experiment succeeded and the harness marked it invalid.** Gameplay reached its full 100-action
budget and stopped normally; the probe then failed because the launcher subprocess returned 74.
Every substantive gate in the card is satisfied. Read the two facts separately.

## The thing that had never happened before: live compression, twice

`integrity.jsonl` records **two** `SUMMARY_INPUT_VERIFIED` events:

| # | history_sha256 | history chars | target chars |
| --- | --- | --- | --- |
| 1 | `23cb42fc…` | 245,368 | 4,000 |
| 2 | `c93bb1b6…` | 218,916 | 4,000 |

The matching summarizer calls, from `model_calls.jsonl`:

| turn | step | prompt tokens | completion | wall |
| --- | --- | --- | --- | --- |
| 17 | 39 | 97,442 | 475 | 53.9 s |
| 29 | 73 | 94,409 | 678 | 57.3 s |

Peak context reached **107,615 prompt tokens**, past the ~102k threshold that no previous run on
this route ever approached. Both summarizer calls finished with `finish_reason: stop`, the guard
verified the exact input serialization both times, and **play continued after each** — compression
at step 39 was followed by steps 40–73, and compression at step 73 by steps 74–100.

This is the first live evidence that the summarization boundary works. Every prior run left it
unexercised: the q6 quick-host and actor-capture runs ended at 10 actions with `summarizer_calls: 0`.

## Gameplay ran to its normal stop

```
stop.json: {"reason": "competition_agent_cleanup", "steps": 100}
official-launcher-result.json: launcher_returncode 0, recorded_action_results 100,
                               grid_hash_changes 99, public_max_actions 100
```

40 solver decisions produced 100 environment steps — the solver batches actions, and the final turn
submitted ten `RIGHT`s at once. 100 transitions recorded, 99 grid changes. `final_state
NOT_FINISHED`, `final_levels_completed 0`, consistent with every chassis on this route.

The rationales show real model-building, not flailing: *"predict: RIGHT -> c_left 44->49, b_left
15->16; burning b-strip toward wrap"*, with an explicit `z` state vector and a versioned hypothesis
(`model v18 (no can_move in z)`).

## Accounting is exact

| quantity | value |
| --- | --- |
| server `chat/completions` accesses | 156 |
| recorded model calls | 155 (153 Actor + 2 summarizer) |
| runtime-owned warmup | 1 |
| unaccounted requests | **0** |
| reasoning tokens | 0 across all 155 |
| finish reasons | 153 `tool_calls`, 2 `stop` |

Containment and teardown clean: `status passed`, `containment_passed true`, `root_returncode 0`,
`term_only true`, 68 teardown entries classified, `forbidden_count 0`.

## The failure: exit code 74 after the work was done

The launcher's own log ends normally — the ARC-AGI-3 `main.py` `cleanup()` path logs
`Received SIGINT, exiting...` (its ordinary end-of-run route, triggered by `os.kill(os.getpid(),
SIGINT)` at `main.py:44`) and calls `sys.exit(0)`. `launcher_returncode` is recorded as **0**.

But the wrapping `run_official_agent.py` process exited **74**, with empty stderr, and
`exact_request_outer_runtime.py:2206` correctly turns any non-zero launcher return into a probe
failure — hence `status: invalid`, `request_path_status: invalid`.

**I could not determine where 74 comes from.** It is not the wrapper's own
`THREAD_FAILURE_EXIT_CODE`, which is 70. It is not produced anywhere in the harness — line 2206
reports the subprocess's real status. `ARC-AGI-3-Agents/main.py` contains one `sys.exit`, and it is
`sys.exit(0)`. The most likely region is interpreter finalization after `SystemExit(0)` is raised
inside a signal handler while game threads are still live, but I have no direct evidence for that
and empty stderr gives nothing to work from. Per the card, this infrastructure failure needs
diagnosis before the run is treated as qualified.

What the evidence does rule out: worker failure (none), hard timeout (3370 s of an 8400 s launcher
envelope and 10800 s session), integrity violation (guard passed twice), reasoning violation (zero
throughout), and missing or extra requests (exact reconciliation). Those are the card's listed
qualification failures, and none of them occurred.

## Disposition

- **The summarization repair is demonstrated live.** Two compressions at ~97k and ~94k tokens,
  verified input serialization, resumed play, exact accounting. That was the question this run
  existed to answer.
- **Qualification is blocked on the exit code, not the boundary.** The card's success wording is
  "clean compression and resumed play with reconciled calls qualifies the live boundary only" — that
  is met on substance, but an infrastructure failure requires diagnosis first.
- **Summary quality remains unmeasured**, exactly as stated before launch. The two summaries were
  475 and 678 completion tokens compressing ~245k and ~219k characters; whether they preserve
  experimental distinctions is a separate question this run does not touch.
- **Zero levels completed.** Reaching the action cap is not gameplay quality, as the card says.

No relaunch was performed. The smallest next step is diagnosing the 74 — capturing the launcher
subprocess's exit path — rather than rerunning three GPU-hours to see it again.
