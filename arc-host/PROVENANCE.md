# Provenance

The ARC host was extracted on repository commit
`27a685f6156039f71b621d958e7645d45cd1eeb6` from two locally exercised host
implementations. It does not import them at runtime.

| Source | SHA-256 | Mechanism retained |
|---|---|---|
| `experiments/harness/runner.py` | `860405e5e2549cad0e99be5c1d74ea631ee083740957cdd141d0ea65acbfd895` | public-frame projection, legality before step, terminal/action boundaries |
| `experiments/harness/types.py` | `971a661a2ec8183cec3f2bd6f4ba0905bb57c9996a47dd9d4e367ec3e6c7788f` | narrow environment/contestant seam and immutable transition vocabulary |
| `experiments/agents/0159-raw-muse-one-turn/run_nooa_ls20.py` | `2b57b989d9a88d2492d8fc79f0b79cfc7609196ba6e81816ffd4b64ab0738b25` | Python 3.14/3.12 process split, prefixed JSONL, persistent worker identity, human action labels |

The extraction deliberately removes experiment cards, model profiles,
provider routing, compaction, E0 projection, and LS20-specific success policy.
Those are candidate or experiment concerns, not ARC-host responsibilities.

`uv.lock` resolves `arc-agi==0.9.9` and `arcengine==0.9.3`. The lockfile is the
executable dependency commitment; the lower bound in `pyproject.toml` expresses
the supported package interface rather than silently substituting another SDK.
