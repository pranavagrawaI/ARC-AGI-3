# Paid production Actor shadow v1

This is a feasibility and resource run, not a decision-quality benchmark. It
uses the actual persistent `TychoNooaAgent` over recorded public observations,
never executes its decisions, and reports historical action agreement only as
an off-policy descriptive measure.

The short six-turn authorization must pass before the separate twenty-turn
authorization becomes usable. Both phases use DeepSeek V4 Flash through the
DeepInfra endpoint only, reasoning enabled, fallbacks disabled, and independent
30-call / $0.50 caps. Combined authorized spend is at most $1.00.

Run from the solver environment after loading `.env` and overriding the frozen
route:

```bash
ARC_LLM_MODEL=deepseek/deepseek-v4-flash-0731 \
ARC_LLM_PROVIDER_ENDPOINT=DeepInfra \
ARC_LLM_CONTEXT_WINDOW=1048576 \
ARC_LLM_MAX_TOKENS=4096 \
TYCHO_NOOA_CAPTURE_REASONING=1 \
TYCHO_SANDBOX_RUNTIME=host \
solvers/tycho_nooa/.venv/bin/python \
analysis/production-actor-paid-shadow-v1/run.py --preflight
```

Remove `--preflight` for the single-use conditional execution.
