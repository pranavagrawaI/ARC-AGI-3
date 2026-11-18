# Reasoning enabled; live qualification incomplete

The requested implementation and tests are complete. Reasoning is established
in actual OpenRouter responses in both arms, not just in configuration flags.
The integrated comparison is not qualified, and no Kaggle launch was performed.

## Implementation and local evidence

Candidate: `deployment/kaggle/designer-player-reasoning-20260907`.
Frozen runtime contract:
`f26e4a95711155fe4420b7dd374c6ecab1f188d0867cc91757960b9b08648d23`.

Both arms explicitly enable reasoning; the actual client now consumes the
16,384-token total output allowance. Reasoning-bearing responses no longer trip
the old recorder abort. Reported token counts and reasoning presence are recorded
separately, without fabricating counts. Exact history and summary-input checks
remain enforced. Empty-content retry is disabled because NOOA otherwise retries
valid reasoning plus tool calls with null prose before parsing their tools.

40 local tests passed, one existing deprecation warning. Tests include actual
worker/subprocess action delivery, both reasoning response fields, disabled and
conflicting configuration rejection, and notebook-extracted execution.
Final `prepare_candidate.py --check` passed. Mocked Kaggle CLI verification passed
with zero network calls, ARC competition binding and RTX PRO 6000 selection.
The notebook's Actor HTTP allowance is 900 seconds; the separately frozen live
pilot used 300-second HTTP, 900-second arms, 48-call safety ceilings and a $3
conservative reservation cap. These are not equivalent hardware/latency tests.

## Live result

| Arm | Admissions / responses / worker records | Reported reasoning tokens | Actions | Levels |
|---|---:|---:|---:|---:|
| loop | 3 / 3 / 3 | 9,904 | 0 | 0 |
| mindset | 16 / 16 / 15 | 46,646 | 1 | 0 |

Reported response cost: $0.369546. Conservative reservations: $1.64493622.
One final mindset response completed after local worker shutdown and was not
delivered; `delivery.jsonl` records it. Every admitted request has a response.
The final relay process exited 0, but that is not a passing experiment verdict.

Loop request 3 finished with `length`: 16,384 completion tokens, 9,721 reported
reasoning tokens, and no tool call. The framework then raised GenerationError.
Headless logged "retrying" but consumed the notification and returned to
`queue_manager.race()`. No action meant no successor notification. This stranded
the operation. An explicit ABORT stopped the arm after the request completed;
460.91 seconds elapsed. This manual diagnostic stop is retained in IPC, not
misrepresented as the automatic deadline or a reasoning-quality failure.

Mindset did submit and execute one DOWN action near the end of its window.
Its prediction check was unresolved: `current prediction is not committed`.
The 900-second outer alarm entered adapter cleanup, which tried to finish while
the worker was still pending. Request 16 was admitted during that cleanup.
This violates the intended admission bound. An explicit ABORT retired local
work; arm cleanup ended at 1,012.42 seconds. The relay waited for the already
admitted request and retained its completed, undelivered response. No retry or
additional paid run was started. Raw evidence and the frozen card are unchanged.

## Decision

Reasoning/tool compatibility is demonstrated, including one real environment
action. Neither the two-action qualification gate nor prediction-feedback
qualification passed. There is no matched reasoning-on/off quality result, no
evidence of improved solve rate, and no target-GPU speed claim.

Keep `launch_ready=false` and `openrouter_qualified=false`. Next repair is bounded
failure handling: a consumed notification must not disappear behind a false
retry log, and a deadline must stop admission before cleanup. Test those paths
locally before another paid run. Do not silently increase the token allowance
or use this failed comparison to reject reasoning or designer-player framing.
