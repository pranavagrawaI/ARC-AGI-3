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

Four interface conventions are documented rather than fixed, because the code that creates
them is inside `participating_source_hashes()` and an edit would invalidate the 0002
release gate. Each was measured defeating a model, not derived from reading — the counts
come from `harness_friction.py` over the 48 recorded 0002 runs.

| Convention | What the model saw before | Recorded failures |
|---|---|---|
| Observation filename width | the glob `observations/step-*.json`; the padding is hidden, and concrete filenames reach the model only via `recent_transitions` — empty at step 0, eight steps deep | **23 of 47** |
| Available packages | nothing; `allowed_distributions` goes only into the host-only manifest | 7 |
| Observation schema keys | "flat objects with top-level public metadata", never enumerated | 4 |
| `sys.path` | nothing; the worker runs `python3 -I` | not separable from imports |

None is advice — all four are facts about the interface. Each is **derived from the frozen
source rather than transcribed**, so it cannot drift: the package list from
`MODEL_RUNTIME_DISTRIBUTIONS`, the schema keys from `observation_metadata`, the filename
width from a format string. P9.1 pins each against reality — it matches the documented
width to the files actually written, imports every named package in the live sandbox,
confirms an unnamed one fails, and compares the documented keys to a real observation file.
Four mutants guard them. Evidence: `artifacts/p9-1-cumulative-loop/evidence.json`.

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

Registered: `connected_components`, which adds `frames.components()` and may move
`model_visible.workspace.provided_helper_sha256` and `model_visible.treatment_capabilities`
— and nothing else. The system prompt is byte-identical between the arms because the
operator documents itself inside the helper; authorising a prompt difference would blunt the
smuggled-prompt negative control. It is registered rather than shipped in the control because
4 of 4 recorded `scipy` imports were `scipy.ndimage.label`, so the demand is real, but
adjacency labelling is a theory of objectness this repository's own findings call
insufficient (LR 0014, 0017, 0019). Denying it does not prevent it — a live model wrote BFS
by hand — so the choice is whether it is visible and measurable or hidden in a dependency
list.

Also registered: `sentinel_probe_note`, which may move
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

## 6. Measuring interface debt

`harness_friction.py` reads any run's `process.jsonl` and classifies each tool failure by
whether the model could have known better. It is the instrument that made §1's conventions
findable: the filename defect was the largest single cause of tool failure across 48 runs
and went unnoticed until someone read a transcript by eye. Baseline over the recorded 0002
runs is committed at `artifacts/p9-harness-friction/recorded-0002-runs.json`.

Attribution is a heuristic over error text, reported as `interface` / `model` / `unclear`.
The counts are evidence; the attribution is a reading of them.

## 7. Running an arm

`run_arm.py` makes 0002-R an experiment rather than a library. It writes the manifest,
trace, workspace snapshot, scorecard, evidence hashes and friction report, and embeds the
arm manifest in each run so pairs can be diffed after the fact.

`--pair` derives both arms from one spec and emits `pair-diff.json`, failing on any
unauthorized difference. Pairs are not assembled by hand: the first one here differed on
`--max-actions` and the diff correctly rejected it. Deriving both arms from one spec makes
that class of error unrepresentable.

Offline mode replays a recording through the fixture policy and shares the live code path
except for the environment and the client, so the runner is covered by the test suite
rather than first exercised against a paid endpoint.

## 8. The freeze

`freeze.py` pins what 0002-R is, so that changing the control has to be deliberate.

`participating_sources()` in `arms.py` digests modules at manifest time and proves the
two arms of one pair executed identical code. That is not a freeze: it is recomputed
every run and no expected value is stored, so an edit to `agent.py` could change the
control while every check stayed green, and a later 0006 result would be measured
against a moving baseline.

Two pinned digests, and they are not independent nets:

| digest | covers | role |
|---|---|---|
| `runtime_digest()` | the eight modules an arm imports, plus 0002's own `participating_source_hashes()` folded in as one entry | the detector |
| `control_manifest_digest()` | every model↔harness crossing of the constructed control | classifies what the detector caught |

The runtime digest is deliberately total — a comment fails it, because nothing
distinguishes a comment from a behavioural change without reading it. The manifest
digest rarely moves alone; it is kept because runtime alone says the control moved,
and the pair says whether it moved somewhere the model can see. Verified: a comment
appended to `agent.py` moves runtime only; editing a sentence of the system prompt in
the same file moves both.

The frozen set excludes the prototypes, the mutation probe, the friction reader and
`freeze.py`. Those measure 0002-R; they are not part of it, and editing a measuring
tool must not read as the control having changed. `test_frozen_modules_are_the_arm_import_graph`
derives the set from `arms.py`'s own imports so the list cannot rot into a hand-kept
fiction.

The pin caught a real defect on its first run: `host_only.source_sha256` globs every p9
module, so the manifest digest included `freeze.py` and changed whenever the pin itself
was re-pinned. It is now dropped from the frozen manifest; code coverage belongs to the
runtime digest, which names its inputs.

**To change 0002-R:** run `python -m …p9.freeze`, read which components moved, and
commit the new constants with a reason. The check runs in the ordinary test suite, so
an undeclared change fails like any other regression.

## What is not frozen here

Budget authority artifacts, a release gate, historical-0002 behavioral equivalence, and
RHAE readiness.

Live runs have been made — Gemma on ls20 and cd82, single runs and pairs — and they
license claims about interface behaviour (adoption, friction kinds, coordinate
submissions, transport failures) at n=1. They do not license any performance claim.
Temperature is 1.0 and the control's own code-cell count varied 63 → 16 between two
identically-configured runs; that variance swamps every between-arm number observed so
far. One level has been completed across roughly 800 actions.

One decision inside the frozen control is unmeasured: the annotated image is supplied
unconditionally, and unlike a helper function it leaves no adoption signal — the model
cannot decline it. The only test is an image-off arm compared across several runs,
which no evidence here provides. It is frozen as a considered guess, not as a
measured one.
