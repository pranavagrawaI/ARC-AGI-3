# Matched Actor Treatment V1

## Question

Does programmable **control** improve frozen ARC judgments once both treatments
already possess the same evidence, pure computations, terminal schemas, and lack
of mutation authority?

## Arms

| Invariant | Native Actor | CodeAct Actor |
|---|---|---|
| Evidence | `evidence_query` | `evidence.query` |
| RLE | `rle_decode_row` | `rle.decode_row` |
| General computation | fresh `run_python(code, values)` | persistent cells |
| Terminal value | task-specific flat `return_result` | same model |
| Writes / ARC actions | none | none |

Native Python receives only explicit JSON values. It cannot query evidence and
its process dies after one tool call. CodeAct may orchestrate evidence and
computation inside persistent Python. This control-language difference is the
intervention.

## Corpus

Twelve reviewed packets from recorded ARC workspaces are frozen into four
balanced cases each:

- **A — computation unnecessary:** machinery can only impose overhead.
- **B — computation useful:** deterministic filtering/equality/order helps.
- **C — composition:** retrieval, transformation, provenance, and judgment must
  remain coherent together.

No compatibility gates or synthetic toy packets enter the cognitive score.

## Measurement

Report the factored evaluator output and the complete resource vector: provider
calls, input/output tokens, dollars, latency, Python cells, disclosures,
invalid interactions, and repairs. Preserve milestone call indices for target
record disclosure, decoding, target identification, provenance binding,
candidate answer formation, and accepted terminal return. Action agreement is
descriptive, not counterfactual correctness.

Milestones are observational: `representation_decoded` requires a completed
cell or accepted decoder call; `candidate_answer_formed` is the first complete
model-proposed terminal payload that the evaluator judges correct, not an
inference about hidden reasoning. `target_value_identified` is therefore the
first call that proves identification, which may be later than cognition itself.

Compare per stratum and by Pareto dominance; do not collapse cost and accuracy
into one score. Both arms receive identical numeric ceilings, not forced equal
actual call counts.

## Zero-cost gate

For every packet, scripted trajectories must disclose identical evidence,
produce identical computation stdout through the shared decoder, return an
evaluator-correct answer, and retain zero canonical-write or ARC authority.
Failure blocks paid execution.
