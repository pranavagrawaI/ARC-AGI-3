# The 0002-R contract frozen by P9

Issue #9 closes only when the candidate works and the executable evidence freezes four
things. Each is stated here with the code and the artifact that establishes it.

## 1. What 0002-R is

0002-R is `build_arm(control_spec(...))`: the unchanged 0002 agent, workspace, transport,
observation transforms, process log, and decision parser, composed with two generic
additions and no ARC-specific cognition.

| Addition | Why it is not optional | Where |
|---|---|---|
| Pointer-only omission notice | #9 requires omission to be explicit rather than silent. 0002 logs `context_eviction` host-side and tells the model nothing. | `conversation.py` |
| Logged transport retry | #9 requires host logs sufficient to reconstruct every request. `OpenAICompatibleChatClient` retries transient failures with no record, so two requests for one round were unreconstructable. | `transport.py` |

Two interface conventions are documented rather than fixed, because the code that creates
them is inside `participating_source_hashes()` and an edit would invalidate the 0002
release gate. Both were observed defeating a model in practice, not derived from reading:

| Convention | What the model saw before | What it does now |
|---|---|---|
| `sys.path` | Nothing. A model-written module simply failed to import — the worker runs `python3 -I`, so the workspace is off `sys.path`. | States `sys.path.insert(0, ".")` before importing your own module. |
| Observation filenames | Only the glob `observations/step-*.json`. The width is hidden; the sole concrete filenames arrive via `recent_transitions`, empty at step 0 and holding just eight steps. Models construct `step-%04d.json` and get `FileNotFoundError`. | States `observations/step-NNNNN.json`, five digits, with a worked example. |

Neither is advice — both are facts about the interface. The filename statement is pinned
against the filenames `ObservationStore` actually writes, so the documentation cannot drift
from the code, and two mutants guard it. Evidence:
`artifacts/p9-1-cumulative-loop/evidence.json`, `findings`.

### Deviation from #9: no authoritative episode record

#9 asks that *authoritative* public episode data remain inspectable after transcript
eviction. **0002-R does not provide that, by decision.** Observations live in the
model-writable workspace exactly as in unmodified 0002; nothing stops the model from
overwriting them, and nothing restores them if it does. P9.2 demonstrates the risk rather
than asserting it away.

A host-owned archive bound read-only at `episode/` was built and satisfied the invariant.
It was removed because its only unique contribution was *authority* — a guarantee the bytes
are the host's. Availability after eviction, which is what the invariant is usually taken
to mean, was never the gap: `ObservationStore` keeps every observation for the whole run,
and recorded runs show 405 reads of those files. Authority had no measured demand: no
recorded run shows a model corrupting its observations. The cost was +816 prompt characters
and +12.4%/+15.2% payload; removing it leaves +330 and +5.5%/+7.6%.

What this forecloses: a 0006 claim of the form "the model reconstructed X from its record"
cannot be defended against "the model wrote that record itself" on this substrate. If 0006
needs that separation it must register the archive as a declared treatment capability, where
the arm diff will show it doing work.

### Deleted: the executed-transition log

An `episode/history.jsonl` mirroring `public_history.jsonl` was built and then removed.
Measured against 8 recorded 0002 runs (414 actions, 165 tool calls, 38–87 evictions each):

- `public_history.jsonl` was read **0 times in 165 tool calls**, despite being named in the
  system prompt and in the `full_history_file` field of every request;
- `observations/` was read heavily — 405 explicit `step-NNNNN` references — but **96% of
  those reference a step still inside the transcript window**; only 4% reach an evicted step;
- the same transition records are already injected inline as `recent_transitions` for the
  last 8 steps, so the file's only unique content was the action taken more than 8 steps back.

The agent never operated at a horizon where the file could contribute. Its unique content —
which action produced which change, long-horizon — is exactly the record a 0006 treatment
should have to earn: the workspace is persistent and a model that wants one can write it.
Re-adding it is therefore a **declared treatment capability**, not substrate. The
`transition-log-reintroduced-into-the-substrate` mutant guards this.

Note the observation files carry no record of the action taken, so beyond 8 steps the
control has frames without causes. That is deliberate.

The host still supplies no frame interpretation, relevance selection, game knowledge,
hypothesis or claim management, prediction requirement, mismatch diagnosis, or semantic
tool routing.

## 2. Substrate shared with 0006

Everything in the manifest outside `treatment_capabilities`. Concretely, both arms share:

- the system prompt (sha256), the tool list, and the observation-message shape;
- the workspace: writable root, read-only projections, reserved paths, output char limit,
  timeout, package allowlist;
- the episode archive schema and mount;
- the context policy: turn limit, python-context policy, omission-notice template digest,
  and the commitment that no host-authored summary exists;
- the failure taxonomy and the decision contract;
- the deliberation budgets;
- the source digests of every participating module.

P9.0 measures this rather than asserting it: 95 of 104 manifest paths and 84/85 and 94/95
observed request-payload paths are identical between the arms.
Evidence: `artifacts/p9-0-shared-arm-seam/evidence.json`.

## 3. Authorized treatment differences

A treatment arm is the control arm plus named capabilities from
`capabilities.TREATMENT_CAPABILITIES`. Each capability declares in advance the manifest
paths and request-payload paths it may move. Anything else is unauthorized.

Always authorized, for any arm: `arm_id`, `host_only.spec.arm_id`,
`host_only.spec.treatment_capabilities`.

Currently registered: `sentinel_probe_note`, which may move
`model_visible.observation_message.payload_keys`,
`model_visible.treatment_capabilities`, and
`messages.*.content.*.observation_payload.treatment`. It exists to prove the verifier
discriminates and carries no evidence, belief, prediction, or routing semantics.

Adding a real 0006 capability means adding a registry entry with its declared paths. A
capability whose declared paths never move is reported as `authorized_unused` — a
capability that changes nothing is a finding, not a pass.

## 4. The smallest verifier

`arm_diff.compare_arms(control, treatment, authorized_paths=...)` followed by
`arm_diff.assert_authorized(report, label=...)`.

It flattens both sides to dotted paths, diffs them, and authorizes a difference only if
the path or an ancestor matches a declared pattern (`*` matches one segment). It runs
over two inputs:

- **declared** — `arm_manifest(arm)`, read from the constructed arm: the live system
  message, the tool list, a probe of the real observation-message builder, the actual
  workspace mounts, and the source digests;
- **observed** — `normalize_request_payload` over the exact payloads each arm sent while
  driven by an identical fixture.

Negative control: P9.0 builds a third arm identical to the treatment except for one extra
prompt sentence, and requires the verifier to reject it on
`model_visible.system_prompt.sha256`.

**Coverage boundary.** The verifier compares arms *against each other*, so it catches
divergence, not drift: a surface added to **both** arms passes cleanly. The shared episode
contract is therefore pinned against a literal in P9.0
(`FROZEN_EPISODE_CONTRACT`) rather than left to the diff. Beyond that, the verifier catches
differences that reach the manifest or the recorded request payloads; a crossing present in
neither is invisible to it. The manifest
is read from constructed objects rather than from `ArmSpec` specifically to shrink that
gap, and `host_only.source_sha256` makes an arm running different code detectable even
when its behavior looks identical.

## Deletion record

The package was put through a delete-centric pass after the contract first went green.
Deleted and kept out because the verifier could not detect their absence and the objective
does not need them: `fixtures.content_turn`, `Checks.inconclusive`, and
`ToolOutcome.model_visible` / `model_repairable` (constants; `failure_taxonomy()` carries
the real claim, and a test asserting a constant is not information). Later, the entire
episode archive and its read-only projection went for the reason in the deviation note
above, taking `episode_archive.py` and `projected_workspace.py` with them.

Survived deletion by failing the verifier: both additions in §1, the source digests, the
capability-registry name check, and the observed request-payload comparison. The last of these initially passed deletion — the verifier
could not see its absence — so the verifier was strengthened instead: the
`treatment-diverges-on-a-path-the-manifest-lacks` mutant diverges the arms on a payload
path the manifest has no entry for, and only the observed comparison catches it.

## 5. The harness-capability verifier

`arm_diff` answers "do the arms differ only where declared". It cannot answer "is the
harness any good", because its checks are the contract restated. `p9_4_harness_task.py`
answers the second question by task completion: a reactive policy maintains an index of
every observation it has seen, and the host grades the result against digests it computed
itself. Seven conditions — clean, eviction, interpreter timeout, retryable transport
failure, oversized output, malformed tool arguments, and all at once — must all produce a
complete and correct index, and each condition asserts that its fault actually fired.

This verifier found the one defect class the contract verifier structurally could not: the
archive copy and the workspace copy of the same observation disagreeing byte-for-byte.
Both are model-visible; disagreement gives the model two incompatible versions of one
public fact with no way to tell. Now checked per condition and guarded by the
`observation-projections-disagree` mutant.

## What is not frozen here

Live endpoint viability, budget authority artifacts, a release gate, historical-0002
behavioral equivalence, and RHAE readiness. None of them is exercised by these prototypes
and none is required by #9's resolution boundary.
