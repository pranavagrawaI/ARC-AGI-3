# Provenance

The build starts from ARC repository commit `f12fca0c1ed8a6fecb22760563fdde7400a9a438` in an isolated worktree. The original dirty worktree was neither cleaned nor used for implementation; its initial porcelain-status SHA-256 is recorded in `source-pins.json`.

NOOA is pinned to `bd51d16e2594f0bc237b14fb31e9a880ec385b39`. The Python 3.12 runtime adapts `InteractiveAgent`, `CodeActStrategy`, sandbox restrictions, and append-only ARC IPC patterns. Its live provider boundary is a byte-for-byte copy of the qualified 0002-NOOA Responses transport at the ARC baseline commit; 0158 adds only the frozen route in its launcher. Its deliberate difference from native NOOA is serialization across the host boundary: there is no live-object pass-by-reference and no model-visible `self`.

Tycho is pinned to `f68912a764372ead0a610db2e1c011d41ce5197e`. The host adapts empirical frame recording, exact `diff_text`, attempt boundaries, and separate terminal/death/animation channels. World-model, outcome, planner, builder, and memory sources are retained only as immutable upstream snapshots for absence/provenance audits; they are not imported into the capability build.

Verbatim upstream snapshots live under `third_party/source-snapshots`. `THIRD_PARTY_LICENSES/Apache-2.0.txt` is copied byte-for-byte from the pinned NOOA checkout. Generated adaptation patches and their hashes are checked by `tools/audit_provenance.py`.
