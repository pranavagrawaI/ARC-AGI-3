# Protocol

V5 changes no cognitive treatment. It retains the V4 six-packet schedule,
Context/Evidence Plane, prompts, caps, authority, and advancement rule.

The only intervention is checkpoint identity. The historical packet field is
preserved as provenance but superseded for V5 admission by `checkpoints.json`.
Its `exclude_python_bytecode_v1` policy hashes every relative file path and byte
except files within `__pycache__` and `.pyc`/`.pyo` bytecode. The V5 loader binds
the old sample and packet bytes, validates the repaired semantic commitment,
and fails closed on any other checkpoint change.
