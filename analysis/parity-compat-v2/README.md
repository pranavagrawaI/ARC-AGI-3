# Parity compatibility v2

This package is the prerequisite ladder for the frozen Native-versus-sterile-CodeAct cognitive comparison. It tests protocol eligibility only; it cannot execute ARC actions or support a reasoning-quality claim.

`gate-manifest.json` binds six immutable gates and the frozen checkpoint. `PROTOCOL.md` defines the branch and stopping rules. `EXPERIMENT-CARD.md` records the bounded precommitment. `experiment.json` remains disabled until Pranav explicitly authorizes a paid route and budget.

The raw representation gate is a diagnostic fallback, not an additional chance after an ordinary cognitive failure. Passing the ladder only makes the model/protocol pairing eligible for a separately authorized shadow experiment.

The trusted launch entry point is `execute_authorized_compatibility_ladder` in `solvers/tycho_nooa/src/tycho_nooa/compatibility_launch.py`. It cannot use the checked-in disabled manifest. A future authorization must bind the exact enabled manifest bytes and a frozen pricing schedule; changing either invalidates that authorization.
