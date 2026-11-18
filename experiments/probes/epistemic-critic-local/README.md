# Local epistemic critic screen

Eight diagnostic cases for a fresh-context critic using the same Qwen3.8-27B-FP8
weights as the Actor. This is a necessary capability screen, not an integrated
Actor–critic treatment or a demonstration of gameplay benefit.

| Pair | Unsupported commitment | Control on identical evidence |
| --- | --- | --- |
| S5I5 scope/objective | Three clicks establish every-click behavior and a counter objective | One further probe, with objective unresolved |
| Search cutoff | Depth-limited failure proves world unreachability | Stop search spending while retaining uncertainty |
| Prediction mismatch | Missing cropped encoding establishes player death | Inspect outside the crop before changing dynamics |
| Initial uncertainty | No action until objective progress is proven | One legal probe with unknown consequences acknowledged |

The first case uses the latest endurance's first three host transitions and exact
turn-3 notes. Its paired proposal is authored. The other six cases are explicitly
synthetic logical tests. `build_cases.py` reads only three transition records;
notes are cut before the next section. Sparse differences retain every changed
cell but omit absolute grids. Therefore this screen measures judgment given those
observations, not independent visual perception. Source path, cutoff and prefix
hash are preserved in `cases.json`; labels and provenance are excluded from model
messages. `rubric.json` is never sent to the model.

## Run offline checks

From the repository root:

```bash
python3 -m unittest discover -s experiments/probes/epistemic-critic-local -v
python3 experiments/probes/epistemic-critic-local/probe.py --model Qwen3.8-27B-FP8 --output /tmp/critic-prepared-01
```

The second command prepares eight requests without network access. Output directories
are exclusive: choose a new name for each attempt. It does not require a model.

## Run the actual model screen

Use an already running local OpenAI-compatible SGLang server serving the exact
Actor checkpoint and FP8 artifact. Copy `EXPERIMENT-CARD.md` to the new run's parent
and record the exact model/artifact identity before running. A server's model name
alone does not verify its weights, quantization, or reasoning configuration.

```bash
python3 experiments/probes/epistemic-critic-local/probe.py --model EXACT_SERVED_MODEL_ID --endpoint http://127.0.0.1:8000/v1 --output /tmp/critic-live-01
```

No automatic model download, server launch, remote provider substitution, retries,
or ARC actions. Eight independent calls, 1024 output tokens each, temperature zero,
thinking disabled in the SGLang request. One model-list request verifies the requested
ID is served. Thirty-second HTTP timeout per completion; first HTTP/protocol failure
aborts and remains recorded. This client timeout does not prove server-side compute
cancellation. Responses with unfinished generation or explicit reasoning content
invalidate the run. Inspect the server configuration for reasoning-off parity too.

## Review outcomes before changing the prompts

Read raw replies against the separate rubric. Record per case:

- Correct disposition and evidence-grounded explanation: pass/fail/unclear.
- Fabricated facts or unjustified claims of certainty: yes/no, with quotation.
- Concrete usable next decision, preserving unresolved alternatives: pass/fail/unclear.
- For controls: unnecessary interruption or demand for certainty: yes/no.
- Latency and tokens from the recorded check/response.

Format validation is automatic; semantic grades remain explicitly pending. A correct
verdict with a wrong explanation fails. An all-challenge critic fails the controls.
An all-allow critic fails the defects. Do not treat an empty `format_errors` list as
metacognitive success. These selected examples do not estimate population accuracy.

If all four defects and four controls pass, proceed to a separate bounded test of
actual Actor uptake with preserved request contexts. Compare critic feedback to an
equal-compute Actor self-review and unchanged continuation. This pack deliberately
does not imitate the Actor with a new generic prompt; it cannot establish uptake.
If any pair fails, inspect the reason before adding orchestration. If responses are
ambiguous, adjudicate the rubric; do not silently relabel after seeing model outputs.

## Local validation result

Seven tests pass, including actual loopback HTTP transport of all eight independent
requests with a scripted server. Offline requests prepared successfully. These are
harness results only. No Qwen inference ran: no local endpoint was configured, and
the inspected local GPU has 8 GB VRAM. No Kaggle launch was performed.
