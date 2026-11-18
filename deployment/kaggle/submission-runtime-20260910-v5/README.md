# Endurance baseline bug fixes — runtime v5

This is the local repair successor to `submission-runtime-20260910-v4`, which
ran in the operationally qualified public endurance v2. The recorded run and
its frozen source remain unchanged.

Use `../submission-20260910-v8/build.py` to package this runtime and
`../submission-20260910-v8/verify_repairs.py` to verify it with the Kaggle Python
environment. The v8 notebook is the entrypoint for this successor. Inherited
contingent-screen notebooks, cards, and preparation scripts describe the donor
experiment and are not a launch plan for these repairs.

Changes: supervise Python cell deadlines from the protocol process, handle grid
shape changes without false no-impact evidence, expose the configured model
context window as client metadata, and reset consecutive refusals after a
successful submission. The context metadata correction does not introduce a
new context-eviction or summarization policy.

This package has no live model or GPU qualification. No upload, game run, or
competition submission is authorized by this directory.

Evidence: `../../../analysis/deployment-probes/20260910-endurance-bugfix/`.
