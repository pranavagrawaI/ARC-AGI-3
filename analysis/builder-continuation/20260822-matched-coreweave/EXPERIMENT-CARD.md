# Matched continuation, pinned route

**Question:** Does retained 8/16 advantage a fresh mechanics Builder over canonical 0/16?

**Contrast:** Source call-491 workspace; only `world_model.py` differs: retained call-492 blob vs canonical. Retained first (seed 20260822).

**Prediction:** Retained ends higher or reaches a shared score sooner.

**Measure:** Exact matched/evaluated before, after verifier results, and final; provider call first reaching score.

**Cap:** CoreWeave `coreweave/bf16`, no fallback; 12 tool steps, 14 calls/arm, 240 s/arm, 28 calls total, $0.42 total with $0.015 reserved before each, 4096 tokens/call, 0 ARC.

**Stop:** Report, timeout, cap, missing cost, provider error, reserve failure, or non-CoreWeave route.

**Invalid if:** source hashes, frontier, endpoint, task, prompt, tools, caps, verifier, differ.

**Result -> action:** retained wins -> add continuation selection; canonical wins -> investigate anchoring; equal -> no benefit claim; invalid -> repair validity only.
