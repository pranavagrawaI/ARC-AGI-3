# Qwen3.6-27B model-switch decision — 2026-08-12

Gemma's live evidence contains repeated failures at the agentic interface:
optional tool avoidance, unsupported narrated inspection, schema-feedback
non-repair, helper shadowing, and repair-budget exhaustion.

The next candidate changes the model to `qwen/qwen3.6-27b`. It retains the
actual prior wire convention `tool_choice=auto`, the evidence and claim APIs,
the verifier, gates, iteration/error budgets, output cap, and one-action host
authority. CoreWeave exposes Qwen3.6-27B as FP8 rather than Gemma's BF16, so the
route is pinned to `coreweave/fp8`; this serving difference is inseparable from
the hosted model switch and is recorded rather than hidden.

Before ARC contact, the exact provider worker must return one typed claim from
synthetic public evidence. Failure prevents approval consumption. Passing that
qualification permits exactly one fresh, non-resumable 44-action LS20 run.
