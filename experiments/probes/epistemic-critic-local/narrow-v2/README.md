# Narrow critic v2: evidence-to-claim audit

Draft instructions: [system.md](system.md). Prepared only; no new inference or
claim of improvement. The first critic and all six original requests/replies remain
unchanged. This prompt revises the task, so its results will not be directly
comparable to the old allow/challenge disposition score.

The first critic judged whether a policy sounded like reasonable exploration and
accepted its supporting story. This version checks claims even when the next action
might be reasonable. It returns at most two quoted claims, evidence references,
specific errors and minimal corrections. It has no action-approval authority and
does not plan a replacement action. Length limits are prompt instructions, not yet
runtime enforcement; future evaluation must count violations explicitly.

## Rule choices

The prompt contains operational checks rather than a catalog of maxims. They cover
literal contradiction, scope, negative evidence, identifiability, causal attribution,
independent validation, objective proxies, parsimony, coverage and belief revision.
Additional maxims should be added only if they detect an error not already covered.
The critic remains responsible for applying the same rules to its own objections.

Corrections to the proposed slogans:

- Absence of evidence CAN support absence when the effect was expected and detectable.
  Otherwise absence is inconclusive. A failed low-sensitivity observation is not a
  universal negative. Research on null-result replication illustrates the distinction:
  [Pawel et al.](https://elifesciences.org/articles/92311).
- Parsimony compares explanations that account for the evidence; it cannot discard
  observations to make an explanation simple. Simplicity and preservation of empirical
  content are discussed by [Scorzato](https://arxiv.org/abs/1402.6664).
- A special case can be observed before a general law is known. A reliable special-case
  counterexample can refute a deterministic universal. What needs justification is
  extending a local observation to untested cases.
- "Something more could be going on" must refer to an unresolved alternative material
  to the inference. Arbitrary possibilities would make every conclusion impossible.
- "All possible evidence" is replaced by coverage of the evidence required for the
  claim's scope. Confounded observations do not isolate effects; see
  [NIST experimental design](https://www.itl.nist.gov/div898/handbook/ppc/section1/ppc136.htm).

These sources motivate distinctions; they do not demonstrate this prompt's efficacy
on ARC. The operational rules and their application here are our design choices.

## Expected exactness on the already inspected failure

A suitable correction to the S5I5 count claim is:

> "Exactly one" is contradicted by transition 2, which changed two cells. The
> observed counts are 1, 2 and 1; no constant increment is established.

A suitable scope correction is:

> Three tested coordinates do not establish every-coordinate behavior. Retain the
> claim that these three clicks changed only row 63; other conditions remain untested.

These are development examples, withheld from system.md and request messages. Since
we used this failure to revise the critic, improvement on it is a regression result,
not held-out generalization. The same packet may contain more than two valid issues;
a concise critic need not list them all. The direct numeric contradiction should
receive priority over weaker inferred concerns.

## Next evaluation contract

For each reported issue, require an exact claim, supplied evidence, a valid inference
objection, and a correction that introduces no unsupported facts. Check that justified
local claims remain untouched. Track omission of direct contradictions, false objections,
and word-limit violations separately. Do not score only the status field.

Use the six old packets for regression first; add held-out decisions before making an
integration judgment. Freeze a new card before any inference. The earlier six-call
allowance was fully consumed. No new paid call or gameplay was made in this edit.
