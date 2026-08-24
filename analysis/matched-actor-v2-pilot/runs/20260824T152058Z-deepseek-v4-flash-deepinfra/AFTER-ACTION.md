# DeepSeek Matched Actor V2-Pilot after-action

## Outcome

**Scientific status: invalid for Native-versus-CodeAct comparison.** The paid
launcher completed all six scheduled gates on the authorized DeepInfra route,
but only the computation-unnecessary control pair completed. The five
cognitive pairs were contaminated by a V2 evidence-adapter type mismatch.

The run made 64 provider calls, spent `$0.005280776`, produced one complete
pair, and executed zero ARC actions. On that sole no-evidence control, Native
was correct and CodeAct was not; this cannot answer the production question.

## Delivered

- `authorization-events.jsonl`: lossless 335-event authorization/provider ledger
- `pairs/`: per-packet arm results, failures, and CodeAct provenance
- `pilot-report.json` and `assessment.json`: frozen evaluator output
- authorization commit: `09f9804`

## Problems

### V1 query object crossed the V2 value boundary

**Symptom.** Every Native cognitive arm repeatedly received:

```text
Input should be a valid dictionary or instance of EvidenceQuery
```

and exhausted six calls. The no-evidence control was unaffected.

**Cause (high confidence).** `MatchedNativeActorController` constructs
`tycho_nooa.evidence.EvidenceQuery` (V1). `matched_actor_experiment_v2` injects
`tycho_nooa.evidence_v2.WorkspaceEvidenceCapability`, whose `isinstance` check
recognizes only the V2 class and then asks Pydantic to validate the foreign V1
model as if it were a dictionary. An offline reproduction fails for the V1
object and succeeds for `query.model_dump()` with the same semantic value.

**Credible alternative rejected.** DeepSeek malformed some requests, but even
its schema-valid Native calls hit the identical host-side model-type error.
This is therefore not principally a model-compliance failure.

**Resolution.** None was applied to this frozen run. V3 should add an explicit
value adapter at the V1-controller/V2-service seam and a real-controller
regression test. The run must not be reinterpreted or rerun under this
authorization.

### CodeAct also failed, for different reasons

CodeAct completed the control and obtained one successful evidence disclosure
elsewhere, but the other trajectories spent their calls on capability discovery,
invalid query construction, or Python errors. Those observations remain useful
interface evidence, but the broken Native comparator and five incomplete pairs
make them insufficient to choose the Actor architecture.

## Decisions

- Preserve the assessment disposition `incomplete_pilot` and classify the
  comparison itself as invalid due to unequal effective evidence access.
- Do not advance CodeAct and do not count the partial 1-0 control as a pilot win.
- Do not issue an adaptive paid rerun. Repair and verify the value boundary first;
  any later execution requires a new manifest and authorization.

## Verification

- Route: all 64 responses reported `DeepInfra` and
  `deepseek/deepseek-v4-flash-0731`; zero route/accounting violations.
- Spend: actual and authorized-accounted spend both `$0.005280776`; violating
  spend `$0`; unknown exposure `$0`.
- Resource safety: maximum preflight input bound `8,848 < 16,384`; maximum
  actual prompt `2,749`; maximum serialized request `8,336` bytes.
- Durable state: 64 planned, dispatched, received, route-validated, and
  accounted calls; six started/completed gates; one completed run.
- Offline causal probe: V1 query object -> V2 validation error; its model-dumped
  value -> valid V2 query.

These checks establish safe execution and the adapter defect. They do not
establish relative cognitive quality.

## Residual and resume

Add an explicit V1-query-to-value adapter, prove Native and CodeAct disclose the
same bounded records through real controller paths, replay one failed packet
without provider calls, run the full tests and CRAP gate, then freeze V3. Only a
freshly authorized V3 pilot can answer the original comparison.
