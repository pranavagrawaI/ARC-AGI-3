# Matched Actor Treatment V1

This package tests whether the Actor needs programmable control, or only
programmable computation behind typed Native capabilities.

- `corpus.json`: balanced A/B/C selection over 12 existing real checkpoints.
- `scripted-audit/summary.json`: zero-cost attainable-world parity result.
- `run_scripted_audit.py`: reproducible zero-cost audit driver; it refuses to
  overwrite an existing evidence directory.
- `experiment.json`: hash-bound, execution-disabled treatment manifest.
- `EXPERIMENT-CARD.md`: immutable paid-run precommitment.

Paid execution remains disabled. Authorization requires a new exact route-bound
manifest; changing the model, prompts, corpus, controller, caps, or pricing is a
new treatment version.

The package deliberately does not reuse the earlier in-memory parity screen.
Before any paid call, this 192-call treatment must be connected to the durable,
hash-bound authorization ledger (or reduced to a separately versioned staged
sample whose caps fit that ledger). The zero-cost runner is not a paid launcher.
