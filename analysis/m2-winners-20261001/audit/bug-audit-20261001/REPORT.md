# First correctness audit — 2026-10-01

Four distinct defects reproduced. Solver sources are unchanged. No GPU, model
server, Kaggle run, submission, or gameplay replay was launched.

This is a bounded audit of the tool/state boundary, not a certification that the
three submissions are bug-free. The strongest immediate repair candidate is
dfranzen's loss of retained helpers on timeout: the mechanism reproduces locally
and the symptom occurs in our completed public-25 control.

## Findings

### B1 — P2: A timeout deletes previously retained functions (dfranzen)

Trigger: retain `def inc(x): return x + 1`, then run `while True: pass` in a later
Python call. The second call neither modifies nor deletes `inc`.

Observed: the sandbox timeout result has no `keepable_functions` field. The
unchanged host `_record_retained_functions` treats that missing snapshot as an
empty successful snapshot and replaces the existing cache with `{}`. An ordinary
`ZeroDivisionError` returns a retention snapshot and preserves `inc`.

Source: [timeout return](../dfranzen-repo/ARC3-Inference/inference/agent/python_tool_sandbox.py#L868),
[host replacement](../dfranzen-repo/ARC3-Inference/inference/agent/tool_agent.py#L4614).
Evidence: `dfr-results.json`, `timeout_erases_preexisting_helpers` and the
ordinary-error control. The reproduction executes the real sandbox subprocess
and the unchanged host retention method extracted by AST.

Gameplay corroboration:

- `runs/out-control/transcripts/su15-1944f8ab_p0.txt:16700`: timeout;
  `:16877`: model explicitly reports that `cheb`, `ringstep`, and `move_center`
  disappeared and starts reconstructing helpers.
- `runs/out-control/transcripts/bp35-0a0ad940_p0.txt:11569`: timeout;
  `:11593`: model reports all retained functions gone.

This does not mean the infrastructure caused those timeouts. For example, the
`bp35` `rowstr` snippet has a `while i < len(s)` loop without advancing `i`.
The additional infrastructure defect is losing unrelated, previously saved code
when the faulty snippet is stopped. Nine timeout result markers occur across
three games in this run; that is not evidence of nine distinct nonempty-cache
wipes or a measured score loss.

Repair criterion: distinguish an unavailable retention snapshot from a confirmed
empty one. An interrupted call should preserve the last committed source pool
with explicit rollback semantics; it must not replay game actions. Whether a
partial new definition executed before interruption is unknown.

### B2 — P2: Model-side mutation can overwrite authoritative action receipts (all three)

Minimal trigger:

```python
receipt = action("UP")
receipt.clear()
```

Observed in each real sandbox runner: the host callback records one executed
action, but the returned `action_results` is `[{}]`. The child appends the same
dictionary it returns to model code, and the host prefers the child's nonempty
list over its own independent execution ledger.

Sources:

- dfranzen: [child alias](../dfranzen-repo/ARC3-Inference/inference/agent/python_tool_sandbox.py#L507),
  [host preference](../dfranzen-repo/ARC3-Inference/inference/agent/python_tool_sandbox.py#L963),
  [caller](../dfranzen-repo/ARC3-Inference/inference/agent/tool_agent.py#L5935).
- lordhansolo: [host preference](../lord-bundle/src/ARC3-Inference/inference/agent/python_tool_sandbox.py#L953),
  [caller](../lord-bundle/src/ARC3-Inference/inference/agent/tool_agent.py#L1127).
- sirikilohit: [active embedded M90 child](m90_wang_sandbox.py#L489),
  [host preference](m90_wang_sandbox.py#L761).

The dfranzen and lordhansolo callers infer `step_executed` from those mutable
receipts, so this result falsely reports no execution and skips normal step
summary updates. For sirikilohit, the corrupted sandbox return is reproduced;
its complete patched outer caller was not reconstructed in this pass.

Evidence: `mutable_action_receipt` in all three result files. A deterministic
callback substitutes for the game engine; it counts actual host calls independently.
No matching real-game incident is established. This is an interface-integrity
bug, not evidence that it explains the leaderboard gap.

Repair criterion: host-owned receipts remain authoritative regardless of what
model code does to returned dictionaries. Preserve intended display behavior
such as dfranzen's removal of `action_echo` from the model-facing return.

### B3 — P2: Retention can silently change a function's meaning (dfranzen)

First call:

```python
sum = 7
def value():
    return sum
print(value())
```

Observed: prints `7`; retention accepts `value` without rejection. The next call
to `value()` returns Python's built-in `sum` function. Retention checks whether
the *name* is provided, without checking whether the live binding still denotes
that provided builtin. The ordinary variable is lost between calls.

Source: [available-name set](../dfranzen-repo/ARC3-Inference/inference/agent/python_tool_sandbox.py#L651),
[dependency validation](../dfranzen-repo/ARC3-Inference/inference/utils/retained_functions.py#L129),
[collection](../dfranzen-repo/ARC3-Inference/inference/utils/retained_functions.py#L284).

Evidence: `shadowed_builtin` in `dfr-results.json`. A control using `offset = 7`
is correctly rejected as an unavailable dependency. Active notebook options were
used: import retention on, repair hints off. A syntactic scan of the public-25
tool calls found no same-call top-level builtin assignment plus function-use
candidate (`builtin-shadow-trace-candidates.json`); this does not exclude every
possible rebinding pattern. No real-game occurrence is claimed.

Repair criterion: reject or explicitly handle dependencies whose live bindings
have been replaced; do not certify a helper under a name-only dependency check.

### B4 — P2: A stale flag falsely says no Python call happened (dfranzen)

Trigger: an exchange with no tool sets `stale_reason='no_tool_call'`; the next
exchange successfully runs `print(2 + 2)` and executes no game action.

Observed: the real tool method returns stdout `4`, but the next prompt still says
“Your previous exchange this turn produced no tool call and executed no actions.”
The non-action branch only updates a summary that is *not already stale*, so the
old reason survives indefinitely until an action replaces the summary.

Sources: [non-action update](../dfranzen-repo/ARC3-Inference/inference/agent/tool_agent.py#L5946),
[prompt rendering](../dfranzen-repo/ARC3-Inference/inference/agent/tool_agent.py#L363),
[no-tool flag writer](../dfranzen-repo/ARC3-Inference/inference/agent/tool_agent.py#L7246).

Evidence: `stale_no_tool_reason_after_real_tool` in `dfr-results.json`. This runs
the complete unchanged `_run_python_tool` method with deterministic observation
adapters and the real sandbox, followed by the unchanged prompt renderer.
In the full control's `su15` transcript, the timeout at line 16700 is followed by
the false no-tool sentence at line 16838. The successful-call probe demonstrates
that the defect is not specific to timeouts.

Repair criterion: describe the immediately preceding exchange accurately while
retaining the older action outcome as historical context. Tool invocation and
game-action execution are separate facts.

## Controls and coverage

Nineteen recorded cases across three runners; assertions check expected behavior,
including the intentionally failing contracts above. This is not a test suite
claiming all production requirements pass.

Controls reproduced:

- ordinary sandbox execution and preservation of host action receipts on a
  subsequent CPU timeout in all three runners;
- dfranzen's rejection of an ordinary unavailable helper dependency;
- dfranzen's preservation of helpers after an ordinary Python exception;
- dfranzen's terminal latch blocks a second action even under `except Exception`;
- lordhansolo's real save/test machinery accepts a valid module and rejects a
  failing replacement; its inspected host commit rule retains old modules on
  errors/timeouts.

Read, without exhaustive testing: dfranzen's context trimming, priority gate,
runtime-state serialization, action batching/guards, and lordhansolo's workspace
and context boundaries. Sirikilohit's final M90 sandbox was extracted from its
notebook; it already contains the M111 animation-view code. Earlier superseded
sandbox patches were not treated as active.

Not qualified: GPU kernels, quantization numerics, speculative decoding equality,
KV/Mamba eviction/recovery, real concurrent scheduling under cancellation,
every game-engine transition, or the full sirikilohit monkey-patch composition.
No claim of zero remaining bugs or causal gameplay improvement is supported.

## Reproduction and provenance

```bash
PYTHONDONTWRITEBYTECODE=1 /home/pranav/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/bin/python3.12 analysis/m2-winners-20261001/audit/bug-audit-20261001/probe.py dfr
PYTHONDONTWRITEBYTECODE=1 /home/pranav/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/bin/python3.12 analysis/m2-winners-20261001/audit/bug-audit-20261001/probe.py lord
PYTHONDONTWRITEBYTECODE=1 /home/pranav/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/bin/python3.12 analysis/m2-winners-20261001/audit/bug-audit-20261001/probe.py siri
```

Python matches the submissions' 3.12.12 runtime. Package initialization that
eagerly imports HTTP/vision is bypassed; the sandbox modules and their own
dependencies are unmodified. Isolated host methods are AST-extracted, not
reimplemented. Test adapters and limits are explicit in `probe.py`.

`manifest.json` records source/evidence hashes. The dfranzen Git snapshot is
`10882e332d883af3e61616df8c8a17d03a328a75`. The **entire** embedded notebook patch
passes `git apply --reverse --check` against that snapshot; this is a read-only
postimage check, recorded in `source-verification.json`. The exported patch needs
a final newline, as IPython's writefile supplies. Lord's source is the archived
bundle at `ca1bd02`; siri's runner is a literal extraction from its saved notebook.

These are the downloaded artifacts under review. This does not independently
prove identity with each final hidden-set scoring submission.

## Decision

Keep diagnosis separate from implementation. B1 and B4 have live-trace
corroboration; prioritize their regression cases. B2 and B3 are latent correctness
failures with deterministic reproductions. Preserve these cases as acceptance
criteria for any later repair. Complete the remaining lifecycle/concurrency and
serving checks before making a broader reliability claim. Do not add a critic or
change the reasoning prompt to compensate for these defects.
