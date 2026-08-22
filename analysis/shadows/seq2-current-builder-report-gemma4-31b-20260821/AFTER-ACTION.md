# After action: sequence-2 builder-report shadow

## Outcome

**Status: partial.** The four-call matched shadow completed without call errors and
contradicted the precommitted immediate-effect prediction: both controls and both
treatments chose `take_action({"action":"UP"})`. Removing only the current
sequence-2 builder report caused no observed change in the first tool call.

The claim is local: at checkpoint `000036`, in these four fresh model samples, the
current report had zero marginal effect on the actor's immediate choice. It does
not establish that earlier builder work, builder-shaped history, or the builder as
a whole has no value. No tool was executed and no gameplay consequence was tested.

## Delivered

- Immutable precommitment: `EXPERIMENT-CARD.md`
- One-leaf contrast and commitments: `perturbation.json`, `packet/manifest.json`
- Raw call records: `result/calls.jsonl`
- Machine assessment: `result/report.json`
- Execution evidence: `preflight.txt`, `execution.log`, `process-exit-code.txt`

## Evidence

| Check | Observation | Establishes |
|---|---|---|
| Changed request leaves | only `/messages/15/content` | intended report-removal contrast |
| Control commitment | equals source request commitment | source request was reconstructed |
| Control repeats | 2/2 `take_action(UP)` | contemporaneous within-arm stability |
| Treatment repeats | 2/2 `take_action(UP)` | contemporaneous within-arm stability |
| Paired first-call change | 0/2 | no observed immediate effect |
| Errors / exit | 0 / process exit 0 | four responses were captured successfully |
| Historical response match | 0/2 | fresh replay did not reproduce the recorded `run_python` choice |

The machine evaluator therefore reports `judgment: contradicted`. The historical
mismatch limits reproducibility: the exact recorded control originally chose
`run_python`, whereas both fresh controls chose `UP`. The matched contemporaneous
contrast remains evidence about marginal immediate effect, but not about stable
response identity across time.

## Problems

### Self-imposed monetary cap was exceeded

Observed cost was **$0.03613761**, exceeding the card's $0.03 cap by **$0.00613761
(20.5%)**. The first control cost $0.02524075; the other calls cost about
$0.0036-$0.0037 each. The four-call and 900-second stops held, but the evaluator
does not mechanically stop on cumulative observed cost. This is a protocol
violation and the reason the overall status is partial.

The first call's upstream prompt cost was $0.02397075 versus about $0.0035 for the
others. No provider-route identifier is preserved, so route drift cannot be
confirmed or excluded. A cold route or provider pricing difference is credible;
ordinary response-length variance does not explain the prompt-cost difference.

## Decisions

Do **not** promote this result to a live game or delete the builder globally. Treat
the current sequence-2 report as a candidate redundant refresh, and mine existing
checkpoints for other places where current-report removal should be tested. Before
another paid shadow, make the monetary stop enforceable and preserve provider-route
identity when the API exposes it.

## Residual uncertainty

The result would be reversed by evidence that the current report changes later
tool use or gameplay despite leaving the first action unchanged, or that an exact
provider-route-matched replication produces arm separation. Earlier builder
reports remain in actor history, so this contrast does not estimate the value of
all builder influence.

## Resume

1. Inspect `result/report.json` and `result/calls.jsonl`.
2. Add a conservative pre-call budget reservation plus cumulative observed-cost
   stop to the shadow executor; test it without provider calls.
3. Use zero-model-cost artifact mining to select the next checkpoint; do not start
   a live game from this result.
