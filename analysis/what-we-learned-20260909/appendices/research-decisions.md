# Research decisions and the earlier acquisition experiments

Scope: coordinator review of experiments 0002–0004, the August promotion/attention ledgers, and the one-Solver consolidation decisions. Results below are historical and model/interface specific. This appendix does not project Gemma results onto Qwen or supplied-route execution onto autonomous discovery.

## The question was already well formed

[Experiment 0002](../../../experiments/hypothesis/0002-decision-relevant-reconstruction/README.md) asks whether a small admissible agent can acquire an executable reconstruction cheaply enough to replace interaction with simulation. It explicitly separates grounding, transition, and objective conjectures. It calls for the smallest reconstruction that changes a consequential decision, not a complete simulator. The README labels itself an architecture and next-loop contract, with no established contestant capability claim.

The later [domain definitions](../../../CONTEXT.md) distinguish machinery, game judgment, reconstruction activation, local content dependence, and beneficial causal use. A model executing code is not by itself evidence that the code influenced its action, and influence alone is not improvement. These are existing distinctions, not a new framework introduced by this report.

## August 7: locating the boundary between selection and execution

Source: [complete 0003 run log](../../../experiments/hypothesis/0003-temporal-game-judgment/RUN-LOG-20260807.md). Exact cards and raw run directories are linked/named there.

| Experiment | Actual result | What survives the result |
| --- | --- | --- |
| Initial A–H ladder | No arm met repeated all-case stability; G/H selected correct options but numeric predictions arrived as strings or shortened keys | Serialization failure must not be called inability to select a plan |
| Repaired H, then typed-record H | 5/6, then 6/6 with the same model/cases and changed serialization | Closed typed transport repaired that interface; no gameplay gain implied |
| Typed-record A–H ladder | A–E 0/6; F 2/6; G 6/6 | Supplied executable options enabled stable selection in this bounded study |
| G perturbation v1 | Reported predictions followed the changed content, but the selected PROBE did not change | Repeating computed facts is weaker than using them to choose |
| G perturbation v2 | Repeated PROBE → ABSTAIN → PROBE after the now nondiscriminating candidate was omitted | Tool/action response dependence demonstrated; strict preferred-reason signature still failed; gameplay benefit not tested |
| Reasoning off/on/off | Off bookends passed; one thinking-on judgment used 2,048 tokens and emitted no tool | Retain reasoning-off for this candidate; no universal conclusion about reasoning |
| Candidate-local layout | All layouts passed; prompt counts fell 3,321→2,340, 4,478→4,018, 4,110→2,604 | A real prior example of cheaper evidence layout with preserved decisions |
| Widened G and E4B transfer | 10/10 primary and 10/10 local E4B over five cases | Transfer of supplied-option selection within one LS20 trajectory, not cross-game discovery |
| E4B reconstruction-disabled control | Invented choices when unavailable options stayed exposed; after authority-sensitive exposure all ten disabled decisions abstained | Offer only executable choices; variable abstention prose does not erase correct action-level behavior |
| One-action online gate | Exact observed successor matched the frozen certificate | Public action delivery and one-step transfer worked |
| Verified-route ceiling | One LS20 level in 13 actions; mean local decision latency 9.53s | Execution with a supplied certified route worked; route discovery was not exercised |
| Full-route ceiling v1 | 375 confirmed transitions, six levels before ARC action-376 transport timeout; INVALID | Strong execution observations, no valid whole-run performance verdict or autonomous-solver claim |
| Full-route ceiling v2 | User stopped after 34 actions/one level | Cancelled partial run, not an additional benchmark |

## August 8: better answers, better grounding, and unsolved routing

Source: [complete 0004 run log](../../../experiments/hypothesis/0004-autonomous-acquisition/RUN-LOG-20260808.md) and [annotation comparison](../../../experiments/hypothesis/0004-autonomous-acquisition/ANNOTATION-COMPARISON.md).

| Experiment | Actual result | Interpretation and disposition |
| --- | --- | --- |
| Objective acquisition v1 | E4B abstained on all three calls among seven compiled scale hypotheses | Safe semantic failure, not transport failure |
| Deduplicated acquisition v2/v3 | v2 lost raw failing payload; v3 selected the correct objective three times after machinery reduced it to one candidate | v2 invalid; v3 reproducible adoption, not independent discovery |
| Route-free planning v1 | 0/3 completions; legal repeated plan; scorer-only route completed checkpoint in 20 actions | Checkpoint feasible, but visible causal affordances incomplete; cannot conclude intrinsic planning incapacity |
| Affordance v2–v6 | Provider, grammar, reference-enum, and output-cap failures | Preserve failed attempts; no pooling them into capability scores; remove redundant constant output fields rather than silently defaulting them |
| Minimal affordance v7 | 0/3 completions; legal 20-action routes never visited rotation trigger or target | Citing an affordance did not make it action-effective |
| Bare-action feedback v11 | 60 legal decisions, 0/3 completions, repeated blocked actions | Per-action feedback alone did not repair the tested policy |
| Annotation audit | Resource exhaustion/reset semantics were missing from model-visible evidence | Retract the stronger claim that the causal package was sufficient; retain zero use of the explicitly supplied trigger |
| Gemma 31B feedback v12 | 60 legal decisions, 0/3 completions, no trigger/target visits | Larger model did not cure this case; provider/precision also changed, so parameter count was not isolated |
| Deliberative v13–v17 | Censored or invalid provider/harness cases | No aggregate quality verdict |
| Deliberative v18 | 60 calls, 0/3 completions; articulated trigger dependency but narrated unobserved positions through blocked moves | Abstract dependency planning present; grounded closed-loop state estimation failed |
| Public-grounded v19/v20 | v19 output-truncated; v20 time-capped after 39 actions, zero tracker mismatches, five blocked actions, no trigger/target visits | Measured state changed recovery behavior; routing/connectivity remained unresolved. Partial evidence, not planned three-episode success |

This sequence directly challenges the idea that a stronger model or more reasoning is sufficient. It also challenges the opposite claim that the model is simply incapable: serialization and grounding changes demonstrably altered behavior. The location of failure moved. No automatic port of these case-specific measurements to a fresh-game contestant follows.

## August 22–25: retained learning and rejected pressure

Sources: [promotion ledger](../../../Questions%20and%20Ideas/Channel%20A%20-%20Promotion%20Ledger.md), [attention register](../../../Questions%20and%20Ideas/Channel%20B%20-%20Attention%20Register.md), [Builder reasoning and matched budget result](../../../Questions%20and%20Ideas/Builder%20Reasoning%20Trace%20-%202026-08-22.md), [Builder CodeAct architecture](../../../Questions%20and%20Ideas/Builder%20CodeAct%20Architecture%20-%202026-08-25.md).

- **Retention and publication are different.** A timed-out candidate scored 8/16 at its contemporaneous frontier while rollback restored a canonical model scoring 0/17 against final evidence. The repaired replay retained the useful candidate without making it canonical or planner-eligible. Final-evidence scoring cannot become a live selector.
- **Continuation preserved a gain, but did not improve it.** Matched CoreWeave continuation finished at retained 8/16 versus canonical 0/16; neither arm gained score. This supports retaining a better starting artifact, not buying more undirected mechanics calls.
- **Goal closure existed already.** Typed-goal replay required an objective review and explicit update before another action. Scripted controls passed. This is protocol evidence; natural uptake and gameplay benefit are separate.
- **A job allocator was rejected for the selected failure.** All 28 calls were mechanics-scoped; mixed jobs were not the cause, and a lower cap would have removed the only positive candidate.
- **Budget pressure was tested and contradicted.** The matched model-visible countdown caused the retained arm to degrade 8/16→0/16 through broad speculative edits; the canonical arm still made no edit. The feature remained research-only, off by default. A fresh suggestion to “make it act sooner” must account for this result.
- **Builder CodeAct was a distinct proposal, not an established improvement.** The document preserves its authority and experiment-parity blockers. Current Actor CodeAct does not retroactively prove this separate Builder treatment.

## August consolidation: an earlier answer to the same circularity

[ADR 0004](../../../docs/adr/0004-separate-solvers-from-experiments.md) diagnosed experiments inheriting implementations, splitting one mechanism across folders, and silent changes to later consumers. It also records that enlarging the 0159–0161 broker did not fix model uptake. The chosen response separated the ARC host, Solver modules, candidate manifests, and experiment records.

[ADR 0005](../../../docs/adr/0005-one-tycho-nooa-solver.md) distinguishes kernel, incumbent, challenger, research, and historical modules. It requires a winning challenger to replace the challenged behavior. Availability is not promotion; dirty worktrees are evidence sources, not automatic merge targets. That historical integration-line decision must not be mistaken for proof that every later Kaggle package imports the same runtime.

[Research disposition](../../../docs/research-lineage-disposition.md) records preservation of a 14,755-path dirty-main corpus in a checksummed recovery archive, rather than bulk promotion. The old `/home/hipra/archives/...` archive path was not present during this review; `/home/pranav/archives` existed but its visible listing was empty. The archive's historic verification is therefore a documentary claim here, not a newly reverified backup. This report inventories the present workspace and does not claim to reconstruct that entire missing corpus.

## Lessons that should not need rediscovery

1. Measure whether the model can select, serialize, execute, and revise separately. A failure at one boundary cannot identify all the others.
2. Preserve gains without granting them authority. Useful rejected/timed-out artifacts need not be lost or published.
3. A repeated experiment is worthwhile only when the candidate, exposure, or decision-relevant uncertainty is explicitly different.
4. Pressure to commit can worsen work; retention/rollback and grounded local edits matter more than urgency alone.
5. Accurate state can improve recovery without solving route search. Neither improvement nor residual failure should erase the other.
6. The existing ledgers and corpora are prior work to reconcile. “Build a corpus,” “add goals,” “use memory,” and “add a planner” are not novel plans without a named missing boundary.

## Coverage and evidence level

The coordinator read the cited reports and decisions, including the later corrections within each run log; raw directories were discovered and checked for presence but every August request was not independently replayed. Reported historical hashes are not newly verified by their appearance in prose. The new source inventory records current document hashes. Related August native Solver results and September packaged behavior belong in the other appendices and must keep their own candidate identity.
