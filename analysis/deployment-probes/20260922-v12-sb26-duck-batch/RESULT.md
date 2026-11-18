# 20260922-v12-sb26-duck-batch result

Kernel `pranavagrawai/tycho-nooa-v12-sb26-duck-batch` v1. COMPLETE 2026-09-22T08:55Z.
Elapsed on the probe clock: 0.012 s. GPU quota moved 7.04 h → 7.05 h.

## Card classification: abort, not a result

`nvidia-smi` returned code 6: `No devices were found`, stderr
`Unable to determine the device handle for GPU0: 0000:05:00.0: Unknown Error`.
Failure stage `accelerator_preflight`. No sessions, no model calls, no actions.

The notebook is COMPLETE only because launcher and assessor use `check=False`.
`ASSESSMENT.json` is INVALID (`runtime not passed`, `no sessions`). That stamp
is correct. Do not read time-to-L1 from this kernel.

Same freeze was not pushed again.
