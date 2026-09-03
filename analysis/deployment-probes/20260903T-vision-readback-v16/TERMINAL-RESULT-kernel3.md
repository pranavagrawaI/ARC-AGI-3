# V16 kernel 3 — invalid run (RTX PRO 6000 not allocated), and it is not our defect

Kernel version **3**, pushed with `--accelerator NvidiaRtxPro6000 --timeout 9000`, launched
2026-09-03, terminal Kaggle status **ERROR** at **~3 min**, same refusal as kernel 2:

```
failure_stage = accelerator_preflight
error         = expected 'RTX PRO 6000' GPU, observed 'Tesla P100-PCIE-16GB'
status        = invalid
elapsed_s     = 0.041
```

## The accelerator flag worked; the hardware was not granted

Unlike kernel 2, the server-side metadata readback now records the requested shape:

```
"machine_shape": "NvidiaRtxPro6000"
```

and `system-before.json` still reports `Tesla P100-PCIE-16GB, 16384 MiB`, `kernel_run_type: Batch`.
Kaggle accepted the request and allocated different hardware.

## Quota is not the constraint

```
resource  used   remaining  total   refreshAt
GPU       9.93h  20.07h     30.00h  2026-09-05T00:00:00
```

Two thirds of the weekly allowance is unspent. This is capacity or availability on Kaggle's side,
not exhaustion on ours, and not a defect in the candidate.

## No fallback exists

The P100 has 16,384 MiB. The pinned checkpoint is 30,866,866,928 weight bytes (~28.7 GB) before
KV cache. The probe cannot be run on the allocated hardware in any degraded form; there is no
smaller-scale variant of this measurement worth taking, because the question is about *this*
route's image preprocessing.

## Cost so far

Kernels 1, 2 and 3 together: no model load, no inference, no environment action, no submission.
The accelerator preflight refuses at ~0.04 s.

## Disposition

Blocked on RTX PRO 6000 availability, not on anything in the candidate. The bytes are frozen and
verified; re-push unchanged when capacity returns. Retry with:

```
kaggle kernels push -p deployment/kaggle/s0-vision-readback-v16-candidate \
  --accelerator NvidiaRtxPro6000 --timeout 9000
```

## Preserved evidence

`remote-source-v16-kernel3/` (metadata readback showing the accepted shape) and
`remote-output-v16-kernel3/` (`result.json`, `events.log`, `system-before.json` with the observed
GPU), downloaded before interpretation.
