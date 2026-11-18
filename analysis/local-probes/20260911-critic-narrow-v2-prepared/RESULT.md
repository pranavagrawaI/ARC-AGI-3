# Narrow epistemic critic v2: rejected after the same six-case screen

The stronger, narrower critic improved selectivity but failed the two decisive
epistemic checks. It should not be integrated in this form.

Six OpenRouter calls completed through CoreWeave FP8 using the same model route,
user evidence packets and inference settings as v1. Only the system instructions
changed. Total request time was 12.162 seconds and cost was $0.0047604. The run
used 13,761 prompt tokens, 340 completion tokens and zero reported reasoning
tokens. All calls completed without retries. No gameplay or Kaggle run occurred.

| Packet | Result | Judgment |
| --- | --- | --- |
| S5I5 actual overclaim | Reported one issue | **Fail.** It caught the unsupported “EVERY click” quantifier, but missed the prior direct contradiction between “exactly one” and transition 2's two changed cells. It also omitted the unvalidated threshold objective. |
| S5I5 cautious control | No issue | Pass. |
| LS20 actual batch | Reported one issue | **Fail.** It invented an absolute band position from sparse change lists and called the Actor's position contradicted. The packet did not supply an absolute grid or measured position for transition 7. It missed the consequential untested bounce premise. |
| LS20 cautious control | No issue | Pass. |
| CD82 actual exploration | No issue | Pass. |
| CD82 authored overclaim | Reported one issue | Pass. It narrowed one observed UP to UP and left other directions untested. |

## What changed relative to v1

The critic no longer judged whether the proposed action sounded reasonable. It
allowed all three cautious controls and produced concise, structured objections.
That is real improvement in task discipline.

The central checking failure remains. Its S5I5 response chose R1 but described a
scope problem rather than checking the literal count contradiction. The prompt
explicitly told it to check counts first and prioritize direct contradiction,
yet it did not. More rules did not make the model perform the required comparison.

The LS20 response demonstrates the corresponding danger: a critic trained to find
an error can manufacture one when the evidence representation is incomplete. It
claimed the band was at rows 47–49 even though the supplied packet contained only
changed cells. That correction violates the critic's own rule against introducing
unsupported facts.

## Decision

The frozen card says to reject the prompt if it misses S5I5's direct contradiction
or makes a false material objection. It did both. Do not integrate or treat this as
a reliable epistemic veto.

The next useful design change is narrower than adding more epistemic maxims: make
the critic first produce a literal observation ledger and compare every numeric or
universal claim against it before any higher-level audit. That mechanism must be
tested separately. No prompt revision or automatic rerun was made after observing
these results.

Evidence is preserved in `00.request.json` through `05.request.json`, matching raw
response and timing files, `STARTED.json`, `EXECUTION.json`, `EXPERIMENT-CARD.md`
and `GRADES.json`.
