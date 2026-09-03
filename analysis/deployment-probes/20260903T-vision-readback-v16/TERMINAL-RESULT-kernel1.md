# V16 kernel 1 — invalid run (packaging defect, no measurement)

Kernel version **1** of `pranavagrawai/prototype-tycho-nooa-vision-readback`, launched
2026-09-03, terminal Kaggle status **ERROR** at **~62 s**. The card's question was not tested.

## What happened

Cell 3's `subprocess.run(..., check=True)` raised `CalledProcessError`. The outer runtime never
started: it exited at argument parsing.

```
vision_outer_runtime.py: error: --r8-l3b requires --startup-timeout-s 1200 --request-timeout-s 180
```

The notebook passed `--request-timeout-s 600`. The R8-L3B lifecycle pins that flag to 180, and the
guard is correct — it protects the frozen one-shot request contract that R8-L3B qualified.

## Why 600 was there

The incumbent `text_think` arm decodes up to 16,000 tokens at ~41 tok/s, about 390 s per request.
Inheriting the pinned 180 s would have timed that arm out and silently deleted the comparison the
probe exists to make. The right fix is a separate ceiling for the probe, not overriding the pinned
one.

## Why the tests did not catch it

Every runtime test parsed an argument list written by hand in the test file, which used 180. None
of them parsed the command the notebook builder actually emits. The candidate was 39/39 green
while the run could not start.

## Cost

No model load, no GPU inference, no environment action, no submission. The SGLang runtime install
completed (`qwen38-fp8-r8l3b-bootstrap/` is in the preserved output); nothing after it ran.

## Repair

1. `--vision-request-timeout-s` (default 600) is now a separate flag used only for the probe
   subprocess; `--request-timeout-s` stays pinned at 180 for the R8-L3B contract. The runtime
   rejects a non-positive value or one exceeding `--vision-probe-timeout-s`.
2. Two tests now extract the `command` list from the **built notebook** by AST and run it through
   `parse_args`. Reintroducing the exact defect makes both fail with the exact Kaggle error;
   reverting makes all 41 pass. The class of bug — a notebook command line no test ever parses —
   is closed.

## Disposition

Per the card's last result-map branch: repair the defect and reissue. The card, frames, conditions,
tasks, scoring, prediction, caps, and claim ceiling are unchanged. `vision_readback_probe.py` and
`frames.json` are byte-identical to kernel 1; only the outer runtime and the notebook's command
line changed.

## Preserved evidence

`remote-source-v16/` (byte-identical readback, confirmed pre-run) and `remote-output-v16/`
(kernel log and runtime bootstrap), both downloaded before interpretation.
