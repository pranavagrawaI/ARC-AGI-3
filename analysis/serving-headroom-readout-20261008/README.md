# Read-only collection

Run: `pranavagrawai/serving-headroom-20261008` v1, kernel 137630249. Frozen tag `serving-headroom-20261008-prelaunch` points to 28724357d6df732fbe29911190f5147e6eed0663. The remote readback uses strings where local notebook sources use line arrays; all seven joined cell sources match exactly. Proof: `../serving-headroom-20261008/evidence/remote-verification-normalized.json`.

`serving-headroom-readout-20261008.service` polls read-only status every 120 seconds, with a 48-hour watch deadline and bounded HTTP timeouts. On COMPLETE/ERROR/CANCELLED it downloads all paginated outputs and runs audit.py. It has no push, relaunch, submission or candidate mutation path. It remains a transient user service: it survives this terminal/turn, but not a machine reboot. No automatic restart or retry of the GPU run.

Inspect: `journalctl --user -u serving-headroom-readout-20261008 --no-pager -n 10`. Actual model hardware, completion and outcomes remain pending while QUEUED. A accepted notebook is not a GPU observation.

Collection validates runtime output hashes, independently recomputes plateau metrics from raw scrapes, checks wire sampler/seed/length/cache receipts, and applies the frozen comparison rule. Profile reduction reports kernel-duration sums and interval union separately, conservatively assigns only explicit marlin_moe/moe_wna16_marlin namespaces, and leaves expert attribution/critical-path interpretation for inspection against actual source. Neither missing profiles nor incomplete blocks become zero throughput.

Local reduction checks passed for independent examples: union([1,3],[2,4],[8,9])=4; two 10us CUDA kernels overlapping by 5us have sum20us and union15us, expert namespace share0.5. CPU annotation spans are excluded. These checks do not qualify native CUDA graph tracing.
