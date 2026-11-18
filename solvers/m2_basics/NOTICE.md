# Sources and modifications

The baseline `src/inference` tree comes from Daniel Franzen's
[ARC-AGI-3 solution](https://github.com/da-fr/arc-agi-3-solution), commit
`10882e332d883af3e61616df8c8a17d03a328a75`, built on Tufa Labs' Duck harness.
The supplied Apache 2.0 license is retained in `LICENSE`. Original source
comments and the notebook's attribution are preserved.

The nonblocking pipe functions and sandbox-cleanup approach are adapted from
lordhansolo's [Milestone 2 source bundle](https://www.kaggle.com/datasets/lordhansolo/taaf-kaggle-source),
archived locally at source revision `ca1bd02`.

Bounded per-action animation retention is inspired by sirikilohit's M111 patch
in [the published notebook](https://www.kaggle.com/code/sirikilohit/arc-agi-3-duck-18-1gc-submit).
The baseline/frame-diff code in that notebook credits Wang Zhongwei/Ludvig;
their embedded runner is retained with its comments in the receipt-only donor fix.

Local changes made 2026-10-01: authoritative action receipts; interrupted-call
retention recovery; builtin binding checks; accurate stale-tool messages;
bounded sandbox transport and cleanup; serialized, bounded animation history;
behavioral regressions and notebook packaging. `candidate.patch` and
`donor_fixes/*.patch` identify the exact source modifications. `manifest.json`
records the source and artifact hashes. These changes have not been evaluated
for gameplay performance or contributed upstream.
