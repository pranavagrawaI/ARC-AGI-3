# Offline qualification

Run from the isolated repository worktree:

```bash
uv run --frozen --with pytest pytest -q -s experiments/agents/0158-nooa-tycho-capability-build/tests
(cd experiments/agents/0158-nooa-tycho-capability-build/nooa-runtime && uv run --frozen pytest -q -s)
uv run --frozen --with ruff ruff check experiments/agents/0158-nooa-tycho-capability-build/{host,tests,tools}
(cd experiments/agents/0158-nooa-tycho-capability-build/nooa-runtime && uv run --frozen --with ruff ruff check agent.py launcher.py tests)
uv run --frozen python experiments/agents/0158-nooa-tycho-capability-build/tools/audit_provenance.py
```

The suite uses fake model cells, fake environment successors, and a sanitized crop derived from preserved public LS20 observations. It exercises the real NOOA sandbox, Python 3.12 subprocess, JSONL IPC, evidence store, verifier, gates, packets, reviews, and live-run guard without provider or ARC contact.

Passing qualification establishes only that the 0158 mechanism behaves as specified offline. Historical replay does not establish Gemma competence, gameplay success, or cross-game generality.

The v25 evidence-conservation qualification adds two regressions: simultaneous primary and outside change remains visible even when the local prediction is supported, and a contradicted static proxy must be followed by a probe-bound model update that cannot label an existing outside residual `none`. The host accepts all three semantic residual classifications and therefore does not choose their meaning.

The v26 provisional-register qualification rejects every non-provisional epistemic status, preserves prior heuristic names and evidence monotonically, permits operational `active`/`watch`/`defer` roles without semantic closure, and exercises omission repair inside the real sandbox before protected-host parsing. It establishes protocol behavior only.

The 2026-08-14 interface qualification contains 48 host tests and 25 Python 3.12 runtime tests. Its recovery regression spends three analysis calls, rereads the current probe on the fourth, submits only one changed heuristic, and verifies that the host retains omitted heuristics and unions their evidence. A paired test proves `max_iterations` still bounds an artifact without a submission. Surface checks require one `inspect_evidence()` entry point, physically omit raw `artifact_input` and the fragmented history helpers, and distinguish fixed-coordinate `predict_cell_values()` from `predict_translate()`. These checks establish interface behavior only; no model competence or gameplay result follows.
