# Flash-Next feasibility run

Version 2 pushed 2026-09-09T11:54Z, status QUEUED. No GPU or model result yet.

Launch: `kaggle kernels push -p deployment/kaggle/flash-next-feasibility-20260909 -t 2700 --accelerator NvidiaRtxPro6000`.
Response: `Kernel version 2 successfully pushed.`

## Version 1 failed before reaching the GPU

Version 1 ended `KernelWorkerStatus.ERROR` after 3.6 seconds, inside papermill, not
in the probe:

    ValueError: No kernel name found in notebook and no override provided.
      papermill/utils.py:52 in nb_kernel_name

The generated notebook carried no `metadata.kernelspec` and no cell `id` fields, so
Kaggle's papermill wrapper refused to pick an execution kernel. This is a notebook
packaging defect and carries no evidence about Flash-Next serving: nothing was
allocated, no setup ran, `raw/flash-next-feasibility-20260909.log` holds only the
nbconvert trace.

`build.py` now emits `kernelspec` (`python3`) plus per-cell ids. Verified before the
v2 push: `probe.ipynb` metadata contains the kernelspec, both cells carry ids, and
the five SHA-256 assertions embedded in the runner cell match `manifest.json` exactly
(`probe.py` 04a87be0..., `actor-cases.json` 55a8224d..., card afd59f1e...,
donor notebook 72d6f351..., donor setup 037c041c...). The v1 failure consumed no
GPU quota and no card allowance; v2 is the first attempt that can produce evidence
under this card.

Scope and immutable bounds: deployment/kaggle/flash-next-feasibility-20260909/EXPERIMENT-CARD.md. The notebook embeds hashed donor source, client, card, and five retained Actor request fixtures; manifest.json records SHA256 identities.

Local qualification: Python compilation of runner and generated notebook; actual loopback HTTP/SSE test proved text/usage/terminal accounting and rejection of Actor prose without an offered tool. These checks do not qualify GPU serving, model quality, or owned-server retirement on Kaggle.

Material source findings: Keith's serving_setup.py pins vLLM 0.1.dev20073+g8e685d198, 32768 context, native MTP 3, and a patched FP8 PLE CPU-offload path. The donor notebook profile disables prefix caching, reserves 5 GiB KV memory, and sets eight maximum sequences. This run preserves that profile; it cannot establish 64K/100K support, prefix reuse, MTP-off performance, or comparative ARC quality.

After terminal status, download all outputs with `kaggle kernels output pranavagrawai/flash-next-feasibility-20260909 -p analysis/deployment-probes/20260909-flash-next-feasibility/raw`. Inspect failure.json, setup.log, setup provenance, each request and SSE stream, results.jsonl, metrics-before/after.prom, resources.jsonl and cleanup.json. A provisional verdict.json pass is insufficient if cleanup or required telemetry failed. No automatic retry.

## The gate over-implements the frozen card (found while v2 was in flight)

Prompted by a peer session's finding that its own assessor had never been run
against evidence shaped like a *successful* run, I ran this probe's gate over
synthetic successful server output on a loopback SSE server
(`gate_test.py`, scratchpad). Result:

| Synthetic successful response | Gate |
| --- | --- |
| content `4`, no reasoning channel | pass |
| content `4`, `reasoning_content` deltas are empty strings | pass |
| content `4`, `reasoning_content` is whitespace `"\n\n"` | **AssertionError: Reasoning-off contract violated** |
| valid `submit_number` tool call, no reasoning | pass |
| valid tool call, `reasoning_content` is whitespace `"\n\n"` | **AssertionError: Reasoning-off contract violated** |

`probe.py:78` asserts `not reasoning` on the raw accumulated deltas. A
thinking-off Qwen3 turn that emits an empty `<think>\n\n</think>` block yields a
truthy whitespace string, so a fully correct response is rejected. Because the
assert lives inside `generate`, which re-raises, this aborts the whole
eight-request run at request 1, forfeits the GPU session, writes `failure.json`
and never writes `verdict.json`.

This abort is not authorized by the card. The card's Stop clause lists: setup
failure, timeout, HTTP/stream error, empty output, malformed or unoffered tool,
failed synthetic exact-output check. Reasoning presence is not among them, and
the card's Measure clause defines TTFT as "first nonempty content/reasoning/tool
delta" — it treats reasoning as a legitimate channel. So the fix is to align the
code to the card, not to amend the card: `assert not reasoning.strip()`.

The donor bounds the risk but does not remove it. `serving_setup.py:2670-2675`
runs the identical thinking-off prompt and requires `content.strip() == "4"`,
and that preflight is known to pass on this pinned stack — so content is
populated and the worst variant (reasoning parser swallowing the whole answer,
tripping the `Empty answer` assert) is ruled out. But the donor never inspects
`reasoning_content`, so whether it is empty or whitespace on this stack is
unobserved. This is a live, unresolved risk for the running v2, not a prediction
of failure.

`jsonschema` is imported inside the tool-validation loop and is absent from the
bare local interpreter, but present in the Kaggle image (`nbformat`, visible in
the v1 log, hard-depends on it) and the probe runs under the notebook's own
interpreter. Not a risk.

Evidence preserved either way: `save(f'{name}.response.json', row)` runs *before*
the asserts, and the raw `{name}.sse` is flushed per line, so an abort still
preserves the complete first response including whatever the reasoning channel
contained. If v2 terminates with `failure.json` naming the reasoning contract,
that is this probe defect, not evidence against Flash-Next compatibility, and
must not be recorded as a compatibility failure under the Result map.

No change has been made to the frozen `probe.py`, `manifest.json` or
`probe.ipynb` while v2 is in flight; their hashes still match what was launched.

## Version 3 pushed 2026-09-09T12:19Z

Authorized by Pranav ("kill e2's v1 and push both corrected candidates"). Not an
automatic relaunch, which the card forbids; the card's Result map directs a new
run once the missing evidence is identified, and it was.

**The card is unchanged.** `EXPERIMENT-CARD.md` still hashes to `afd59f1e…`, the
Actor fixtures to `55a8224d…`, and both donor files to `72d6f351…` /
`037c041c…`. Only `probe.py` moved: `04a87be0…` → `2dac7930…`. All three fixes
align code to the frozen card rather than amending it.

| Line | v2 | v3 | Card authority for the change |
| --- | --- | --- | --- |
| 84 | `assert not reasoning` | `assert not reasoning.strip()` | reasoning presence is not on the Stop list |
| 100 | `assert calls and finish == 'tool_calls'` | `row['actor_returned_tool'] = …` | every Actor case sends `tool_choice: "auto"`, so prose is permitted by the request |
| 138 | prompt "the single key value" | names the key `"value"` explicitly | the gate is card-stated; the case was written ambiguously |

The donor assets were re-verified before the rebuild rather than trusted:
`/tmp/flash-next-keith-20260909` no longer existed, so it was reconstructed from
the v2 kernel output and both files hash-matched the frozen manifest exactly.
`kernel-metadata.json` is byte-identical after the rebuild, so the docker image,
dataset, model and machine pins are untouched.

**The gate was run over passing evidence before the push, not after.** Ten
synthetic cases through a loopback SSE server (`gate_test_v3.py`, scratchpad):

- Four valid runs that v2 would have aborted — whitespace `reasoning_content`,
  a correct `{"value": 4}`, an Actor answering in prose under `tool_choice:
  "auto"`, an Actor returning a valid `execute_python` call — all accepted.
- Six card-stated stops — wrong `json` value (the exact v2 abort), wrong `text`
  content, empty output, unoffered tool name, schema-violating arguments,
  missing terminal usage — all still rejected.
- `actor_returned_tool` recorded `False` on prose and `True` on the tool call,
  which is the measurement that replaces the abort.

10/10. One case first reported as a failure was the test's fault, not the
probe's: it sent `{}` to a tool whose schema requires `code`, and the probe
correctly rejected it.
