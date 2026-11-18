# q6-tool-use-contact

Written 2026-09-04, before launch. q5 S1 = 0 measured discoverability.
This run puts the helper in the cell so the failure we observe is a use
failure.

Question: With `symmetry.py` imported in `execute_python` and its source
in the prompt, which use-failures occur when this actor is asked to find
similar or congruent figures on a frame that has a known pair?

Posture: diagnosis. One factor: the helper is in hand. Not a game. Not
S2 (betting matching is the win). Not discoverability.

Contrast: two frames, same model, helper, prompt, temperature.
- A: ar25 L0 turn 0 — 4-connected same-colour CCs recover the pair
- B: ls20 L1 step 32 — those CCs split the pair; a colour-9 window
  recovers it

The model prompt names neither game nor pair. Scoring uses an oracle
that is not sent.

Prediction: U1 ≥ 1 on both (the source is in the cell). U2 (target pair
returns True) is more likely on A than on B.

Measure:
- U1 = wrapped calls to `symmetry.{similar,congruent,directly_congruent,canon,transporter,orbit,primitive}` ≥ 1
- U2-A = a call whose two figures are n=45 colour {4} and n=45 colour {11} and the result is True (or a nonempty transporter)
- U2-B = a call whose two figures are n=24 colour {9} and n=6 colour {9} and `similar` is True (scale is required; `congruent` is False on this pair)
- Other True pairs recorded, not gated
- Extraction method read off executed source (CC / window / other)

Cap: 8 `execute_python` rounds per trial, 15 s per cell, **$0.12**
OpenRouter, 20 min wall. Remaining $0.16 of today's $10 daily limit.
`qwen/qwen3.8-27b`, thinking off, temperature 0. Seed digest
`7e0c1f1cdcd0d594…` (10579 B).

Stop: a trial ends on a non-tool reply, the round cap, or remaining
credit < $0.02. The run ends after both trials, or skips B if remaining
< $0.04 before it starts. Abort on spend ≥ $0.12 or `reasoning_tokens > 0`.

Invalid if: `reasoning_tokens > 0`; game name, pair location, or oracle
sizes in the model prompt; GRID absent from the interpreter; imported
digest ≠ `7e0c1f1c…`.

Result map:
| result | next |
|---|---|
| **U1 = 0** | They will not call the module even when it is in the cell. Do not spend a game run on `read_helper` in the workflow. Next is a different uptake path or a stronger model — own card, after credit reset. |
| **U1 ≥ 1, U2-A = 0** | The API is reached; a CC-visible pair is missed. Failure is quotient or figure construction. Inspect wrapped calls. Do not run ls20 as a game. |
| **U1 ≥ 1, U2-A = 1, U2-B = 0** | Extraction method is the live failure. Next is an extractor in the seed, or ar25 as the live game — own card, after credit reset. |
| **U2-A = 1 and U2-B = 1** | Tool-use is not the bottleneck. Next is S2 in a game run — own card, after credit reset. |
| **Invalid** | Do not interpret. New id. |

Outputs: `analysis/local-probes/20260904T-symmetry-tool-use/results/`
