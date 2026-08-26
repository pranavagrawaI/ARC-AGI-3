# Builder proposal freeze audit

## Decision

The Builder proposal prototype is frozen as a nonconformant experiment. It is not a live Solver treatment, must not execute hostile generated code, and must not be used for a paid Native-versus-CodeAct replay.

The freeze preserves the proposed protocol, adapters, sterile execution experiment, matched-replay scaffold, and focused tests so later work can resume from inspectable evidence. It does not certify the implementation against `BUILDER_PROPOSAL_PROTOCOL.md`.

Actor-only work may proceed while this experiment remains frozen.

## What the current checks establish

- The focused proposal and adapter suite passes 21 tests.
- The complete Tycho Nooa Solver suite passes 631 tests when run with the repository's trusted local-development sandbox mode, `TYCHO_SANDBOX_RUNTIME=host`.
- Ruff reports no violations in the Solver package.
- The live `TriggerBuilderLoop` does not construct a proposal session or invoke either proposal adapter. Legacy callers still use `TheoryTransaction.finish()` without proposal arguments.

These checks establish narrow value, publication, adapter, and scripted replay paths. They do not discharge the protocol's ordered authority, recovery, conformance, or paid-replay obligations.

## Known blockers

### Publication and recovery

1. `TheoryTransaction._finish_proposals()` checks canonical bytes before publication but does not hold a lock or perform a compare-and-swap when replacing `world_model.py`. Newer authority can appear after the stale-context check and be overwritten.
2. Multiple publishable candidates are classified as ambiguous before an eligible report nomination is considered. This contradicts the protocol's selection order and does not compute maximal candidates.
3. Transaction recovery trusts a matching `canonical.json` record to choose a blob without revalidating the publication case, report, receipt, lineage, or decision. This weakness predates the proposal path, but it prevents the proposal implementation from claiming PO-9 recovery.
4. Recovery tests do not interrupt every durable publication boundary or cover corruption and the stale-check/publication window.

### Realm authority

1. Generated Builder Python can write directly to inherited protocol descriptors. A raw write to stdout can forge a frame and desynchronize the parent, so the current process does not have one typed egress.
2. The authority tests cover one unauthorized capability call and one immutable mapping mutation. They do not cover the structural attack set required by the protocol.
3. The Builder worker monkey-patches private members of `_sterile_worker.py`, and the Builder executor copies its process-launch lifecycle. Isolation therefore depends on private implementation details.

### Protocol representation

1. Every `EvidenceSnapshot` is labeled `host_observation`, although the source frontier does not distinguish environment evidence from model-authored notes or hypotheses.
2. Obligation definitions omit evidence/oracle provenance and per-obligation comparison rules.
3. The implemented evaluation receipt shape and status vocabulary differ from the protocol's exact wire schema.
4. The Builder realm sends raw parent and candidate source; the host constructs the parent-bound patch. The realm therefore does not emit the protocol's exact `CandidateSubmission` value.
5. Evaluator identity is accepted as an arbitrary `evaluator_sha256` string rather than derived from executable code, dependencies, configuration, seeds, locale, and other semantic inputs.
6. `FrozenBuildCase.builder_view()` omits committed model ABI and trusted dependency information needed to interpret the incumbent.
7. Protocol-level reports permit weaker validation than adapter-return reports, and caller metadata can overwrite transaction-owned `cause` and `focus` fields.

### Matched experiment validity

1. The replay hashes caller-declared provider settings but does not prove that both LLM clients use them.
2. It does not commit common and adapter-specific prompts, a shared arm deadline, or continuation policy.
3. It does not exercise a common transaction/publication policy.
4. Native `run_python` values can replace the trusted `case` input, breaking arm parity.
5. The scripted fake-client replay proves only that both adapters can be made to return one known candidate. It does not establish model, provider, budget, prompt, or execution parity.

## Conditions for resuming Builder work

Before a paid replay or live treatment:

1. Decide whether the normative protocol or the Python wire values change where they currently disagree.
2. Close the publication race and define committed candidate-selection policy.
3. Repair and test recovery at every durable boundary.
4. Remove raw protocol descriptors from generated-code authority and complete the structural attack suite.
5. Preserve truthful evidence provenance and complete obligation semantics.
6. Bind evaluator identity to every semantic input.
7. Make matched-arm configuration mechanically enforced rather than caller-asserted.
8. Run the protocol's required test sequence in order.

Until then, this checkpoint is evidence about a design attempt, not an implementation ready for promotion.
