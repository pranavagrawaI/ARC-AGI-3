# 0003 invalidation record

- **Decision date:** 2026-07-21
- **Status:** frozen invalidated prototype
- **Superseded by:** `../0004-counterexample-gated-ledger/`

## Licensed result

0003 is a negative control demonstrating that controller-invoked executable
API compliance can masquerade as prospective reconstruction.

Its checker accepts implementations that return an action echo while learning
nothing. Candidate calls can also communicate through shared mutable runtime
state. The reveal gate does not prove exact legal-action coverage, commitment
identity, isolated evaluation, or successful post-reveal learning.

Therefore 0003 licenses **L0 executable API compliance only**. It does not
license claims of action-conditioned prediction, reconstruction, learning,
planning adequacy, or gameplay benefit.

## Freeze rule

Do not repair the implementation or reinterpret historical results under new
criteria. Corrections belong to a new experiment so that this failed construct
remains inspectable. Documentation may point to the successor, but the 0003
source, prompts, tests, and acceptance predicate are historical evidence.