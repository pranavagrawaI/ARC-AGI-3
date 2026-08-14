# 0158 implementation report

Date: 2026-08-11  
Branch: `experiment/0158-nooa-tycho-grounding`  
Baseline: `f12fca0c1ed8a6fecb22760563fdde7400a9a438`

## Result

The G0-G2 capability build is implemented and qualified offline. It provides immutable public evidence, host-issued claim and commitment identities, one-action execution, primary-region-anchored effects, non-oracular verdicts, bounded promotion, permanently terminating live failures, and future-run-only human review.

The build deliberately exposes no G3+ module or API. It uses NOOA's agent/runtime structure across serialized process boundaries, not live-object pass-by-reference.

## Verification

| Surface | Result |
|---|---|
| 0158 host behavior | 27 passed |
| 0158 Python 3.12 NOOA runtime | 3 passed; one multiprocessing deprecation warning |
| Inherited 0002 host regression | 36 passed; 21 subtests passed |
| Inherited 0002 runtime regression | 92 passed; two multiprocessing deprecation warnings |
| Host and runtime Ruff checks | passed |
| Host and runtime byte compilation | passed |
| Pinned-source, patch, card, and contract audit | passed |

The offline historical shadow exercised six real ephemeral NOOA artifact processes and reached G2 using three linked, prospectively committed, exact translations. It did not contact a model provider or ARC service.

One falsifying probe found that replacing the CodeAct prompt wholesale also removed the sandbox's repair feedback. The runtime now rebuilds the fixed contract and artifact task while retaining only the within-artifact assistant/tool/error turns. The sandbox tests demonstrate that repair feedback remains available while `self`, framework objects, host paths, network access, and non-allowlisted imports remain unavailable.

## Claim boundary

This qualification establishes mechanism behavior only. It does not establish Gemma grounding competence, LS20 level completion, gameplay capability, or cross-game generality. At qualification time the frozen card recorded `execution_authority: false`; a later separate approval authorized exactly one run. That run is preserved in `docs/runs/LIVE-RUN-20260811.md` and did not change the implementation claim. Success remains `levels_completed >= 1`.
