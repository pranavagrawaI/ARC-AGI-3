# P9 — the shared arm seam (issue #9)

Five small prototypes, not a harness rewrite. Each one answers a question, writes
its evidence to `artifacts/<name>/evidence.json`, and fails loudly.

`llm_repl_agent/*.py` and `experiments/harness/{runner,trace,types}.py` are inside
`experiment_protocol.participating_source_hashes()`. Editing any of them invalidates the
0002 release gate, so nothing here modifies them. Every addition composes: a
`RollingConversation` subclass, an `LlmReplAgent` subclass, and a transport wrapper.

## Running them

```
UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl-cache uv run python3 \
  experiments/agents/0006-capability-environment/p9/p9_0_shared_arm_seam.py
```

…and likewise `p9_1_cumulative_loop.py`, `p9_2_context_loss.py`,
`p9_3_failure_recovery.py`. All four also run inside the suite:

```
UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl-cache uv run python3 -m unittest \
  experiments.agents.tests.test_p9_shared_arm
```

No provider, API key, or ARC endpoint is involved. The model is a fixed script and the
environment replays frames from `recordings/`.

## The prototypes

| Prototype | Question | Result |
|---|---|---|
| **P9.0** `p9_0_shared_arm_seam.py` | Do two arms from one factory differ only where declared, and does the verifier catch an undeclared difference? | 95/104 declared manifest paths shared; the 9 that differ are all declared by `sentinel_probe_note`. Driving both arms through the same fixture, 84/85 and 94/95 observed request-payload paths are identical and the single difference is the sentinel field. A smuggled prompt change is rejected on `model_visible.system_prompt.sha256`. |
| **P9.1** `p9_1_cumulative_loop.py` | Does inspect → author → receive → reuse → act hold over three recorded observations? | Holds. A helper module and notebook written at step 0 are reused at step 1 against an observation that did not exist when they were written, and the notebook accumulates. It also carries the two interface-convention findings — `sys.path` and the observation filename width — and pins the documented filename against the files actually written. |
| **P9.2** `p9_2_context_loss.py` | What survives forced eviction? | Every observation, including evicted steps, stays readable; so do the model's own files. The omission is stated as a pointer-only notice that grows with the evicted set, and no other host-authored message reaches the model. It also measures the accepted cost: those observation files are model-writable and nothing restores them. |
| **P9.4** `p9_4_harness_task.py` | Does the harness carry a **real task** to a correct answer under injected faults? | Yes, in all seven conditions. A reactive policy maintains `index.json` over every observation seen; the host grades it against digests it re-derives from the recorded frames, independent of any component under test. Eviction (3), interpreter timeout with restart, retryable transport failure, output truncation, malformed tool arguments, and all of them at once each leave the index complete and correct. |
| **P9.3** `p9_3_failure_recovery.py` | Are the failure modes distinguishable, correctly routed, and retried only when safe? | Five tool signatures are distinct; syntax and runtime failures share a kind but name different exceptions. The timeout restarts the interpreter, loses in-memory state, keeps files and the read-only record, and says so. The illegal action returns to the model and is repaired. The retryable transport failure is retried once, logged twice host-side, and leaves no trace in any request the model saw; the fatal one is not retried. |

## Interface conventions the model had to guess

Two facts about the harness were invisible and defeated a model in practice. Neither is
fixable in code — both come from `participating_source_hashes()` — so both are stated in
the contract and verified:

- **`sys.path`.** The worker runs `python3 -I`, so a model-written module will not import
  without `sys.path.insert(0, ".")`.
- **Observation filenames.** The model was shown only the glob `observations/step-*.json`.
  The zero-padding width is exactly the part that must be guessed, and the only concrete
  filenames arrive via `recent_transitions` — empty at step 0, eight steps deep. Models
  construct `step-%04d.json` and get `FileNotFoundError`. Now stated as
  `observations/step-NNNNN.json` with a worked example, **pinned against the filenames
  `ObservationStore` actually writes** so it cannot drift, and guarded by two mutants.

The generalisable part: a glob in a contract hides precisely the thing the model has to
construct.

## Testing the tests

A self-verifying prototype that passes proves nothing unless a prototype that *should* fail
does. `mutation_probe.py` breaks one named property at a time and requires something to
notice:

```
UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl-cache uv run python3 \
  experiments/agents/0006-capability-environment/p9/mutation_probe.py
```

Current result: 14/14 caught (`artifacts/p9-mutation-probe/results.json`). It edits source
in place, restores in `finally`, and verifies every digest afterwards, so it is deliberately
outside the test suite.

### Three defects in the probe itself

Running it adversarially found more wrong with the probe than with the code it tests:

1. **A stale patch counted as a survivor.** After the chain digest was deleted, one
   mutant no longer matched any source text and was silently scored as surviving.
   That both understates coverage and hides that the probe stopped testing anything.
   Stale is now a hard failure.
2. **No concurrency lock.** Two runs restore from each other's stale snapshots and can
   leave mutated source behind; a test run alongside one fails for no real reason.
   It now refuses to start while another holds `.mutation-probe.lock`.
3. **It overwrote the evidence it exists to protect.** The probe runs the prototypes
   against broken source, and `Checks.finish` writes `evidence.json` *before* it
   raises — so every probe run replaced the committed artifacts with mutant output.
   At one point `p9-2-context-loss/evidence.json` on disk claimed the prototype
   failed. Prototypes now honour `P9_ARTIFACTS_DIR` and the probe points it at a
   scratch directory.

The first run of this probe found two survivors, and both were real:

- `archive-verify-always-reports-match` survived **everything**. Both P9.1 and P9.2 asserted
  that the archive verifies and nothing ever showed `verify()` returning false — a
  vacuous check. Fixed by `EpisodeArchiveTests`, which corrupts and deletes observations
  and requires the mismatched record to be named.
- `timeout-does-not-report-restart` survived P9.3, which read `workspace_restarted` off the
  raw record instead of the classifier. P9.3 now asserts through `classify_tool_result`.

Division of labour: the prototypes exercise properties end to end; the unit tests cover the
negative half that an end-to-end run cannot reach (a corrupted archive, an exhausted retry
budget, a rejected projection). `archive-verify-always-reports-match` is still caught only
by the unit tests, and that is the intended split.

## Why P9.4 exists

P9.0–P9.3 assert that the arm matches its declared contract, which can only ever
confirm the contract: the checks encode the requirement, so a requirement-level
question cannot be put to them. P9.4 measures something the contract does not
mention — whether a policy pursuing one objective across an episode ends up with the
right answer. A harness weakness surfaces as a wrong answer rather than a failed
assertion.

Its fixture is a **policy, not a script**: it reads the conversation it is handed and
decides from the last tool result, so it can be defeated. Its ground truth is re-derived
by the host from the recorded frames, never read back from a component under test.

The escalation earned its keep before the archive was removed: it found that the two
model-visible copies of one observation could disagree byte-for-byte, which passed every
prototype, test and mutant at the time. That defect class no longer exists — there is
only one copy now.

## Modules

| File | Role |
|---|---|
| `arms.py` | `ArmSpec`, `build_arm`, `arm_manifest`. The only construction path for either arm. |
| `capabilities.py` | The treatment capability registry. Each entry declares the paths it may move. |
| `arm_diff.py` | Normalized flatten/diff/authorize, used for both manifests and request payloads. |
| `agent.py` | `SharedArmAgent`: `LlmReplAgent` plus the omission notice and capability hooks. |
| `conversation.py` | `RollingConversation` plus the pointer-only omission notice. |
| `transport.py` | Retry loop moved out of the client so each attempt is logged. |
| `failures.py` | The failure taxonomy and its classifiers over unchanged 0002 behavior. |
| `fixtures.py` | Scripted model client and recorded environment. |
| `task_policy.py` | Reactive policy with a real objective, and the host's ground truth for it. |
| `evidence.py` | Check recording and evidence reports. |
| `mutation_probe.py` | Breaks one property at a time and requires the suite to catch it. |

The frozen contract these establish is in [`CONTRACT.md`](CONTRACT.md).

## What this evidence does not license

- No claim about gameplay, learning, or any 0006 intervention. The model is a script.
- No claim that the manifest covers every model↔harness crossing. The verifier catches
  differences that reach the manifest or the recorded request payloads; a crossing absent
  from both is invisible to it. It also compares arms to each other, so a surface added to
  *both* passes — the shared episode contract is pinned against a literal in P9.0 instead.
- No claim of live readiness: no provider, real endpoint, budget authority, release gate,
  or RHAE evaluation is exercised.
- No security claim. `ProjectedWorkspace` makes the episode record read-only to the
  sandboxed model; it does not make the sandbox a hostile-code boundary.
