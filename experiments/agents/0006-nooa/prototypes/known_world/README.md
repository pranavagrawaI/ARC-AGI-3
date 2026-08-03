# Known-world LS20 prototype

Question: can the frozen `0002-NOOA` control beat LS20 level 1 within 44 live actions when
rule inference is removed?

The three independently approved conditions are:

1. `rules`: a complete tutorial in `RULES.md`, without a level route.
2. `source`: readable access to the exact obfuscated `ls20.py`.
3. `executable`: the same source plus an executable exact world-model bundle.

Every assisted run stops when level 1 is beaten or after 44 live actions. The executable
bundle is differential-checked against the authoritative game before approval. This is a
throwaway capability probe: record the answer in `NOTES.md`, then delete or absorb it.

Prepare all three no-spend approval requests from the repository root with one command:

```bash
uv run python experiments/agents/0006-nooa/prototypes/known_world/prepare_approvals.py
```

Live execution still requires the runner's digest-bound confirmation and API credential.
