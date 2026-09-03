# V16 kernel 2 — invalid run (wrong accelerator allocated, no measurement)

Kernel version **2**, launched 2026-09-03, terminal Kaggle status **ERROR** at **~3 min**.
The card's question was not tested. The kernel-1 repair held: argument parsing succeeded, the
outer runtime started, and it failed later and for a different reason.

## What happened

```
failure_stage          = accelerator_preflight
failure_classification = pre_request_failure:accelerator_preflight:RuntimeError
error                  = expected 'RTX PRO 6000' GPU, observed 'Tesla P100-PCIE-16GB'
status                 = invalid
elapsed_s              = 0.038
```

Kaggle allocated a Tesla P100. The runtime's accelerator preflight refused before touching the
checkpoint, so no model load and no inference occurred. `result.json` records
`claim_ceiling = "The run failed and establishes no deployment feasibility claim."`

## Cause

`kernel-metadata.json` declares `machine_shape: NvidiaRtxPro6000`, and the byte-identical remote
readback confirmed the push carried it. That is **not sufficient**: the Kaggle CLI writes
`ApiSaveKernelRequest.machine_shape` from the push-time `--accelerator` flag, so a plain
`kaggle kernels push -p <dir>` leaves the shape unset and Kaggle allocates its default GPU.

This project already knew that. Board thread #124 reply #129 records the same trap from R5/R6 and
its fix: *"both omitted an explicit batch machine_shape and Kaggle allocated Tesla P100 … official
enum is `NvidiaRtxPro6000`"*, resolved in R7 with
`kernels push --timeout 1800 --accelerator NvidiaRtxPro6000`. Kernels 1 and 2 were pushed without
the flag because the metadata file appeared to declare it.

## What this run does establish

Only that the guard works. The preflight is doing exactly its job: it converted a silently wrong
substrate into a refused, classified, zero-cost failure rather than a P100 run whose numbers would
have looked like evidence.

## Repair

Push with `--accelerator NvidiaRtxPro6000 --timeout 9000`. The timeout covers 1,200 s startup plus
the 5,400 s probe deadline with headroom, and caps the run if anything hangs. No candidate bytes
change: the notebook, probe, runtime, and frames are byte-identical to kernel 2.

## Cost

No model load, no GPU inference, no environment action, no submission.

## Preserved evidence

`remote-source-v16-kernel2/` (byte-identical readback, confirmed pre-run) and
`remote-output-v16-kernel2/` (kernel log, `result.json`, `events.log`, `probe-config.json`,
`backend-facts.json`, `shutdown.json`, system before/after), downloaded before interpretation.
