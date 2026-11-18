# Claim-by-claim critic: implemented and tested

The same-model checker failed the direct contradiction even when given the exact
claim to check. Selection is therefore not the only demonstrated problem.

Twelve requests completed, comprising six individual-claim checks, two three-claim
list checks, two extraction calls and two checks of those extracted lists. Total
cost was $0.0100134 and summed request time 41.779 seconds, with 9,406 input tokens,
2,345 output tokens and zero reported reasoning tokens. All model requests used the
same Qwen/CoreWeave FP8 route, temperature zero and 1024 output-token cap. No gameplay.

## The decisive result

The claim was the Actor's exact wording:

> action = any CLICK x y -> exactly one bottom-row cell (next from right) turns 3->4.

The supplied host measurements explicitly gave changed-cell counts 1, 2, 1, plus
the exact changed coordinates and values. In the individual-claim condition the
model returned supported_in_scope and stated:

> All three transitions show exactly one bottom-row cell changing from 3 to 4,
> matching the claim.

The list condition made the same error. Extraction then preserved the exact-one
claim verbatim, but its verifier returned unresolved because only three clicks
had been observed. That is a different epistemic error: one reliable counterexample
refutes the deterministic universal in its stated conditions; it is not necessary
to test all clicks before rejecting exactly-one.

| Target | Expected | Individual | List | Extract then verify |
| --- | --- | --- | --- | --- |
| S5I5 exactly one changed cell | Contradicted | Supported: wrong | Supported: wrong | Unresolved: wrong |
| S5I5 current count four | Supported in scope | Correct | Correct | Correct |
| S5I5 threshold completes level | Unresolved | Correct | Correct | Correct |
| CD82 red count/bounds | Supported in scope | Correct | Correct | Correct |
| CD82 white bounds | Supported in scope | Correct | Correct | Correct |
| CD82 border change might be counter/player | Mechanism unresolved | Supported only the measured change; incomplete | Correctly unresolved | Mechanism omitted by extraction |

Expected-label agreement is 4/6 for individuals and 5/6 for lists. The sixth target
combines an observed change with a modal mechanism hypothesis, so its individual
grade is mixed rather than an unambiguous false assertion about the mechanism.
Extraction retained 5/6 target propositions; four retained targets were correctly
classified and exactly-one remained wrong. Do not treat these tiny selected counts
as estimates of general accuracy or rank the conditions from one sample.

## What the implementation tests

`claim_screen.py` separates extraction from verification. Extraction sees original
Actor prose only and returns exact excerpts. A shared verifier handles every claim
in a supplied list, with one required judgment per ID. Verification facts are
computed from the saved prefix: changed counts, current color counts and bounds,
and color counts in affected rows. No raw-frame mental counting is required.
The same facts and verifier instructions are used across the three conditions.
Extracted claims are passed verbatim, without manual repair or adding omissions.

All supplied claims were assigned judgments. That establishes list coverage at
the interface, not sound reasoning. The CD82 extraction dropped the phrase about a
possible counter/player indicator, showing that exact copying alone does not
ensure proposition coverage. An earlier annotation or host measurement cannot
automatically certify that the selected claim set is complete.

Sixteen local checks passed, including prefix exclusion, literal measurement
oracles, paired-evidence equality, label isolation and lossless code-fence parsing.
AUDIT.json verifies frozen inputs, identical evidence across conditions, exact
extraction transmission, response ID coverage, cost and original-response retention.
Three citations used `coverage` or `note` field names rather than formal evidence
IDs; these are recorded format findings, not evidence of model correctness.

## Operational interruption, preserved

The first attempt at ../20260911-critic-claims-v4-run1 stopped after one completed
response because the strict JSON parser rejected a markdown code fence. That reply
already contained the semantic failure quoted above. Its card, error and raw
response remain unchanged.

A separate continuation card admitted eleven remaining requests. The parser now
accepts one enclosing JSON code fence without editing its payload. The original
response was copied byte-for-byte and scored once, never requested again. Across
both records exactly twelve paid calls occurred, all within the original $0.25 cap.
Eight replies used code fences. This was a documented continuation, not an
uninterrupted operational pass or an extra attempt at the failed claim.

## Judgment

This rejects the hypothesis that claim selection was the sole problem. Verification
still failed in this evidence packet after the disputed claim was explicitly named.
It does not prove the model cannot compare three bare numbers in a different task;
the effect of distracting facts and the general reasoning configuration remain open.

The useful capability is narrower: the checker correctly handled the supplied
current measurements and kept the objective hypothesis unresolved. It should not
be the sole authority for factual verification. Numeric, coordinate and formal
predicate checks should be executed when expressible, while semantic criticism
remains provisional and evidence-referenced. No further prompt run or architecture
integration followed this result.

## Reproduce

- Local tests: `python3 -m unittest discover -s experiments/probes/epistemic-critic-local -q`
- Read-only source/reply audit (writes AUDIT.json):
  `python3 experiments/probes/epistemic-critic-local/audit_claim_screen.py --output analysis/local-probes/20260911-critic-claims-v4-continuation`
- See DATA.json for frozen expectations and provenance, PROMPTS.json for both roles,
  GRADES.json for manual assessment, and named request/response files for every call.
