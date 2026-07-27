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

## Running an arm

`run_arm.py` is `run.py`'s shape minus the 0002-specific authority, freeze and
release-gate machinery: manifest, trace, workspace snapshot, scorecard, evidence hashes,
and a friction report.

```
# offline: no provider, no endpoint, no API key. Replays a recording through the
# fixture policy. This is the path the test suite exercises.
uv run python3 .../run_arm.py --offline --max-actions 4

# a matched pair, with the arm diff emitted as pair-diff.json
uv run python3 .../run_arm.py --offline --pair --max-actions 4

# live
uv run python3 .../run_arm.py --game ls20 --model <id> --max-actions 40
```

Live and offline share one code path; only the environment and the client differ. Offline
exists so the runner is tested before a provider is involved — an untested runner is how a
live run gets wasted.

`--pair` derives both arms from one spec, so they cannot differ except by capability. That
is not a convenience: the first hand-assembled pair here differed on `--max-actions`, and
the arm diff correctly rejected it. A pair assembled by hand measures the operator.

Each run's `manifest.json` embeds its `arm_manifest`, so two runs can be diffed after the
fact without rebuilding either arm.

## What live runs found

Five live `ls20` runs against `google/gemma-4-31b-it`, read through `harness_friction.py`
and the workaround detector. Each defect below was found by running, not by inspection,
and none would have been caught by a check written in advance — the check would have
encoded the same assumption the contract did.

| found | how it surfaced | fixed by |
|---|---|---|
| model believed it had no per-step images | a hedge in code it wrote: `# Since I don't have the files explicitly` — **no error, nothing logged** | stating the rendered image path |
| filename separator, not width | `step00000.json`, `step_00036.json` — width always right, separator wrong twice | stating the literal Python expression, pinned by evaluating it |
| a whitespace response body killed a run | provider returned 1001 bytes of whitespace; the frozen client marks a JSON decode failure fatal; run died at action 12 of 44 | retrying malformed bodies in our wrapper, and stopping the episode instead of destroying it |

Three blind spots, three different instruments. The workaround was invisible to the friction
log (`ok=True`). The transport death was invisible to it too (it reads tool failures, and
reported `tool_failures: 0` on a run that had just been killed). No single instrument sees
the whole surface — which is the argument for the play-test probe.

After the fixes: 68 tool calls in one run with zero filename failures, and two transport
failures retried without touching the episode.

## The play-test

`--probe` gives the harness to a model as its first user. Run against `moonshotai/kimi-k3`
on ls20: 12 actions, 35 Python calls, $2.38. It kept the running record it was asked to keep
and filed seven claims. Every one was checked against the request payloads before being
acted on:

| claim | verdict |
|---|---|
| "workspace starts empty" is false — five entries exist at step 0 | **real**; the frozen prompt says otherwise |
| `recent_transitions` is capped at 8, undocumented | **real**; found by trial at step 9 |
| digests are not reproducible from the files | **real**; undocumented |
| the PNG is an exact 8× upscale of the grid and adds no information | **real**; see below |
| deliberation counters are stale and unreliable | **false — confabulated** |
| `public_history.jsonl` is missing at step 0 | **real, now fixed** |
| observation keys match the contract exactly | confirms an earlier finding was a test artifact |

The false claim is the most useful result in the table. It was the most alarming line in the
report — "counter semantics unclear/unreliable, treat with care" — and the payloads show the
counters are exactly right at every round. That is why the interview is quarantined as
hypotheses: **the scariest thing a tester tells you may be the one thing that is not true.**

Probing costs the prober. Kimi spent its repair budget deliberately sending `"2"`, then a
missing `action` key, then `2.0`, and terminated its own run at action 5. Probe mode now
allows 20 repairs: a repair cap protects a contestant, but in a play-test it forbids testing
the surface the play-test exists to test.

**Open, not fixed:** if the rendered PNG really is an exact upscale of `pixels[]`, every
request pays for an image carrying no information the grid lacks — and images dominate the
token bill (2.16M prompt tokens for 44 actions). Dropping it is an arm difference, not a
contract fix, so it belongs in the capability registry as a measurable experiment rather
than a silent change.

## Pure probe vs play-along

Both ran `moonshotai/kimi-k3` on ls20 for 12 actions, sharing `ARC_AGI_3_BRIEF` verbatim so
the difference is attributable to the role. **They found almost disjoint sets.**

| pure probe — hunts the interface | play-along — plays for real |
|---|---|
| workspace not actually empty | transcript omission destroyed hard-won conclusions |
| `recent_transitions` capped at 8 | **the layer count changes mid-run** |
| digests not reproducible | request envelope duplicates everything three times |
| PNG is a redundant 8× upscale | no last-action-effect field; no player position; no numeric timer |

A hunter enumerates what is *discoverable*. A player finds what *bites*. The play-along
report's opening line — "nothing crashed, but two things functionally broke my play" — is a
sentence a pure prober cannot write, because nothing was at stake for it.

### The finding that justifies the whole exercise

The play-along probe noticed that at step 9 its observation went from 1 layer to 6 and back.
Verified, then checked against the historical corpus:

- **40 of 648** recorded observations are multi-layer (2, 5, 6, 13, 15, 22 layers);
- **404 of 591** recorded tool calls index `pixels[0]`;
- `pixels_to_png` renders `pixels[-1]`.

So on those steps a model parses a stale layer while the image it is shown displays a
different one. No exception, no friction-log entry, no way to notice — the code simply
computes on the wrong grid. This sat in 48 runs of evidence and no instrument could have
found it, because it never fails.

The contract now states that the layer count varies and the current state is `pixels[-1]`,
derived from the renderer and pinned against `representation_descriptor`.

### What the player asked for

Unprompted, its `worst_single_thing` was: *"give the agent a pinned, harness-preserved
notes/summary block that survives transcript omission."* That is 0006's Context Assembler,
requested by a model that had never heard of 0006 — and arrived at by losing its own
conclusions to eviction and having to re-derive them from files.

It also reported re-measuring a room it had recorded before the drop and finding its own
earlier note wrong: *"after a transcript drop, nothing I 'remember' should be trusted."*

## Fixes taken from the probes, and what was refused

Triage test for every request: **does granting it lower the cost of using the interface, or
lower the difficulty of the task?** The first is harness work. The second is the experiment,
and refusing it is correct.

Taken:

| fix | evidence |
|---|---|
| duplicated payload mirrors removed | 5 keys appeared twice per turn — **47% of the observation payload** |
| `pixels[-1]`; layer count varies | 40/648 observations multi-layer while 404/591 tool calls index `pixels[0]` |
| `NOTES.md` echoed verbatim every request | a player lost its conclusions to eviction and named this the single worst thing |
| workspace isn't empty; transitions capped at 8; digests opaque; history file exists at step 0 | each found by a probe, each pinned |

Refused, all for the same reason — they lower task difficulty, not interface cost:
the controllable object's bounding box (a game-specific prior `CONTEXT.md` forbids), a
semantic effect report, a numeric timer readout, batched moves.

Net: 0002-R is **15.9% cheaper** than unmodified 0002 while carrying six documented
conventions and a persistent notes channel. It started 12.4% more expensive.

## The notes channel collected nothing on its first run

`--harness-notes` asks for one-line `WANT:` / `FRICTION:` / `WASTE:` notes while the model
works. First live run: **zero notes.** The model wrote a `notes.md` full of game analysis
and never touched `harness_notes.md`.

The cause was the prompt, not the model. `PLAY_ALONG_ROLE` asked for "a running record in
your workspace — a file you append to"; the notes convention asked for lines in
`harness_notes.md`. Two instructions about note-keeping, one file created, zero notes
collected. The model reasonably merged them and used the result for the game.

Both now point at `NOTES.md` — the file the harness already echoes back in full every
request. The model sees its own notes and the prefix format every turn instead of recalling
one line from the tail of a long prompt, and only prefixed lines are read as feedback.
Whether that fixes it is untested: it is a hypothesis, and the next play-along run is the
test.

## Measuring where the harness defeats the model

`harness_friction.py` reads a run's `process.jsonl` and classifies every tool failure by
whether the model could have known better from what the harness told it:

```
UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl-cache uv run python3 \
  experiments/agents/0006-capability-environment/p9/harness_friction.py
```

Over the 48 recorded 0002 runs — 595 actions, 593 tool calls, 47 failures — **34 of 47 are
attributable to the interface** (`artifacts/p9-harness-friction/recorded-0002-runs.json`):

| count | kind |
|---|---|
| **23** | `constructed_observation_path_not_found` — `observations/step-0007.json` |
| 11 | `python_error` |
| 7 | `module_unavailable` — `No module named 'sklearn'` |
| 4 | `observation_schema_key_guessed` — `KeyError: 'current'` |
| 2 | `tool_argument_validation` |

It reads only `process.jsonl`, so it works on runs that predate every module here and on
any future run. Attribution is a heuristic over error text and is labelled as such.

## Interface conventions the model had to guess

Four facts about the harness were invisible. None is fixable in code — they all come from
`participating_source_hashes()` — so each is stated in the contract, **derived from the
frozen source so it cannot drift**, and pinned against the live sandbox:

| Convention | What the model saw | Recorded failures |
|---|---|---|
| Observation filename width | the glob `observations/step-*.json` only | **23** |
| Available packages | nothing; the allowlist is host-only | 7 |
| Observation schema keys | "flat objects with top-level public metadata" | 4 |
| `sys.path` | nothing; the worker runs `python3 -I` | — |

Each is derived rather than transcribed: the package list comes from
`MODEL_RUNTIME_DISTRIBUTIONS`, the schema keys from `observation_metadata`, the filename
width from a format string P9.1 checks against the files actually written. P9.1 also
imports every named package in the live sandbox and confirms an unnamed one fails.

The generalisable part: **a glob in a contract hides precisely the thing the model has to
construct**, and anything the model must build a name for — a path, an import, a key — is a
rediscovery cost until it is stated concretely.

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
| `harness_friction.py` | Reads any run's process log and reports where the interface, not the game, defeated the model. |
| `run_arm.py` | Runs one arm, or a matched pair, live or offline. |
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
