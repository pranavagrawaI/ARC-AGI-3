# Solver modules

`solvers/` contains stable implementations. Experiments select and configure a
solver; they do not contain or inherit its source.

The environment-side runtime is separately buildable in
[`/arc-host`](../arc-host/README.md). Solvers implement that host's narrow
persistent-process protocol rather than importing the ARC SDK.

| Module | Behavioral target |
|---|---|
| [`tycho_nooa`](tycho_nooa/README.md) | Tycho's native workspace-tool loop inside NOOA's agent/strategy architecture |

Boundary rules are recorded in
[`docs/adr/0004-separate-solvers-from-experiments.md`](../docs/adr/0004-separate-solvers-from-experiments.md).
