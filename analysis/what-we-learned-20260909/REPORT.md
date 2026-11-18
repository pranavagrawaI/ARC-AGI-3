# What we have learned about our ARC-AGI-3 solver

Evidence reconciliation, 9 September 2026. This is a synthesis of preserved research, not a new solver proposal or a claim of current leaderboard performance.

## Read this first

We have built working interactive solvers, completed levels, recovered useful computational artifacts, and demonstrated several narrow improvements. We have also repeatedly mistaken implementation for exposure, exposure for benefit, an invalid experiment for a capability failure, and an old candidate's weakness for the current system's state.

The work is not empty. The evidence does not yet establish a reliably strong, fast, general contestant. Both statements must survive the report.

The repeated cycle is partly a retention failure in our own research process. The same ideas recur under different names and in different runtime families. Some were already implemented, some were tested and rejected, some were never exercised by the model, and some worked under supplied conditions that the model has not learned to produce. Their dispositions are not interchangeable.

The most useful unit of retained understanding is therefore:

> On this exact candidate, under these conditions, this capability was available; this behavior actually occurred; this outcome followed; this comparison does or does not establish a cause; this is what we retain.

## Navigation and evidence coverage

| Read | Purpose |
| --- | --- |
| This report | Integrated conclusions, corrections, repeated ideas, and remaining uncertainties |
| [Research decisions and acquisition](appendices/research-decisions.md) | August 7–8 selection/grounding experiments; promotion, retention, and consolidation decisions |
| [Early Solver history](appendices/early-history.md) | Native Solver, goal/model/tool experiments, parity work, and historical successes |
| [Runtime and deployment](appendices/runtime.md) | Actual execution, host parity, lifecycle, q2–q6, and gameplay evidence |
| [Inference and performance](appendices/inference.md) | FP8/MTP, sampling compatibility, concurrency, prefix cache, and Flash-Next |
| [Discovery and procedural learning](appendices/discovery.md) | Designer-player, reasoning, demonstrations, transfer, and self-test experiments |
| [Controller and experiment checks](appendices/controller.md) | Supplied-plan execution, live adoption, level completion, and withdrawn comparisons |
| [Source inventory](sources/source-documents.json) | Paths, sizes, dates, and SHA-256 identities of 4,489 discovered Markdown documents |
| [Board snapshot](sources/board-log.json) | All 385 delivered board events through cursor 385; unrelated threads remain identifiable |
| [Snapshot metadata](sources/snapshot.json) | Checkout identity, initial dirty state, capture time, and collection limits |

Five subagents investigated disjoint research slices. The coordinator reviewed earlier acquisition/decision records, reconciled cross-slice conclusions, and checked selected decisive artifacts. The source inventory is broader than the documents substantively adjudicated: it is not a claim that every document or raw model response was read. Existing raw data remains at its original locations. We did not rerun paid experiments or rewrite old result files.

The old consolidation record names an additional 14,755-path recovery archive under `/home/hipra/archives`. That old path was unavailable in this review; the visible `/home/pranav/archives` listing was empty. Its historic verification remains a documentary claim, not a newly verified recovery guarantee. See [research disposition](../../docs/research-lineage-disposition.md).

## 1. There is no single undifferentiated “our solver”

| Family | What it is | What must not transfer automatically |
| --- | --- | --- |
| Early machinery/Gemma acquisition experiments | Supplied options, public measurements, route ceilings, and bounded semantic decisions | Their successful supplied-route execution is not autonomous Qwen discovery |
| Unified native Tycho-NOOA | Persistent Actor with native workspace tools, optional Builder, goal/planner/transaction machinery | A module or test in this family does not establish access from the packaged NOOA CodeAct Actor |
| Kaggle NOOA join and q-series | Adapter and worker around the pinned NOOA Actor, with candidate-specific host and feedback changes | A q-number or copied source snapshot does not establish the loaded runtime or launch status |
| Designer-player/discovery successors | Changed decision interface, experiment records, interruption, demonstrations and self-tests | Local interface tests and prompt inclusion do not establish model adoption or gameplay benefit |
| Contingent-controller successor | Bounded first action plus observation-conditioned continuation | A supplied-plan local win does not show the model will propose such a plan |
| Serving candidates | FP8/MTP/cache/configuration/Flash-Next experiments | Serving synthetic requests or replaying tool calls does not establish complete Solver compatibility or score |

[ADR 0004](../../docs/adr/0004-separate-solvers-from-experiments.md) and [ADR 0005](../../docs/adr/0005-one-tycho-nooa-solver.md) already separated source, candidate, experiment, incumbent, challenger, and historical material. That distinction was a response to earlier circularity. This report applies it to later evidence rather than inventing a new lineage architecture.

## 2. What the evidence does establish

### Execution is real; reliability is conditional

The official adapter can start/reset games, obtain model decisions, execute legal actions, reconcile exact successors, and close workers. The q6 summary-boundary candidate ran 100 actions on LS20, changed the grid 99 times, and performed two live summary calls. The official launcher returned zero. A stale post-run validator assigned exit 74 because it expected the donor identity. That is materially different from “the solver did not run.” It is also not permission to rewrite the original experiment verdict.

Later packaged runs also reached their action caps. The contingent-controller v3 arms each executed 20 actions; one completed a level. These observations defeat the blanket claim that nothing runs or no level has ever been solved. They do not establish reliability across new games or repeated seeds. See [runtime evidence](appendices/runtime.md) and [controller evidence](appendices/controller.md).

The earlier native NOOA q2d run also recorded one level, 24 submissions and 123 actions; its turn-based procedural gate passed even though an event-based scorer called it failed. q3c completed LS20's first level in 32 actions and wrote/read a consolidation note, then defended an incorrect next-level goal despite accurate visited-path dynamics. That is concrete evidence of both capability and its limit. q4 subsequently executed 197 actions without reaching the goal-transfer boundary, so its goal treatment remained unexposed. These local runs are distinct from the Kaggle joins.

### Level completion must count even when treatment attribution fails

The September 9 `on` arm completed one level in 20 actions, with 878.9 seconds of gameplay. The `off` arm completed zero in 20 actions, with 1,134.5 seconds. No Actor contingent proposal or continuation occurred in the `on` arm. The completion is a real solver outcome; it is not evidence that the unused controller caused it. One game/seed without replication cannot establish a reliable advantage.

There are older successes too. Two August symmetry-assisted native Solver runs each completed one ar25 level on the same seed. The Actor's comparison and planned action suffix are visible in the retained accounts, even though the shared outcome/planner machinery did not close. Those are positive same-seed replications, not a cross-game result or a matched symmetry ablation. The original raw symmetry directories are absent here, so these counts are attributed to the preserved reports.

The August E4B verified-route ceiling completed one level in 13 actions, but it was supplied a certified historical route. The larger ceiling run observed six levels before an operationally invalid transport stop. Such execution ceilings must be preserved with their supplied-information conditions, not merged into autonomous performance totals. [Earlier experiments](appendices/research-decisions.md) and [native history](appendices/early-history.md) retain the outcomes and their evidence limits.

### The model sometimes knows more than its action reveals

Early Gemma experiments selected correct supplied plans while failing to serialize nested predictions. Typed records repaired that boundary. Later deliberative runs articulated the trigger dependency but advanced an imagined position through blocked moves. Public-derived position measurements changed recovery behavior without solving route search. In Builder traces, a concrete local repair hypothesis existed before the deadline, but no edit followed.

These observations rule out a universal diagnosis of “insufficient intelligence.” They identify different boundaries: serialization, grounding, commitment, route search, and objective selection. They do not prove that all remaining failures are interface problems. [Detailed acquisition and Builder evidence](appendices/research-decisions.md).

### Time is mostly spent crossing the model boundary

In the preserved q6 100-action trace, gameplay took 3,005.16 seconds; summed Actor request time was 2,979.78 seconds across 153 Actor calls. There were two summarizer calls, zero levels, and approximately 28.4 effective output tokens/second. That effective rate includes request work; it is not a pure decode-kernel benchmark. Actor and summarizer durations must not be stacked as disjoint wall-time shares: their sum exceeds gameplay wall time.

Once a valid batch exists, host execution is inexpensive. Much of the delay occurs while the Actor reads evidence, writes/tests helpers, repairs mistakes, and decides what to submit. Faster inference and fewer useful-decision crossings are distinct levers. Neither can be inferred from raw actions-per-call alone: the batch eligibility correction shows why. [Performance evidence](appendices/inference.md).

### Useful artifacts can survive without becoming authoritative

A timed-out Builder candidate scored 8/16 at its evidence frontier. Rollback restored a worse canonical model. The repaired retention path preserved the candidate while keeping it noncanonical and planner-ineligible. A subsequent matched continuation retained the 8/16 versus 0/16 difference but improved neither arm.

The gain is preservation of useful work. It is not proof that more generic Builder calls will improve it, or that a retained candidate is safe to publish. This distinction is already implemented and tested in the historical native family. [Promotion record](../../Questions%20and%20Ideas/Channel%20A%20-%20Promotion%20Ledger.md).

### Memory and interface work produced real, bounded gains

The older structured-retrieval comparison improved exact older-evidence answers from 12,12,11/15 with flat retrieval to 15,15,15/15 with structured views, while using about 69% fewer prompt tokens and 62% less reported cost. Live compaction recovery also preserved the partial turn and exactly one subsequent action. These are retrieval and recovery gains; no corresponding game-score improvement was established. Typed-goal proposal, independent review, selection to revision 2, and subsequent delivery were exercised in a later live native run. It is wrong to describe all of that as unimplemented or merely scripted. [Early-history appendix](appendices/early-history.md).

### Procedural transfer progressed beyond the original handoff

| Later test | What actually happened | What we learned |
| --- | --- | --- |
| Literal two-cell feasibility | Two submissions in 2+2 Actor calls, evidence-responsive revision after no-op; second action not applied before diagnostic cleanup | The stripped procedure can be followed; no paired no-demo effect or second-transition evidence |
| Five-case reliability | Four cases passed, but the mandatory competing-regions case failed; all used UP then DOWN | Overall reliability gate failed despite real evidence consumption; prospective discrimination and flexible action choice remain separate |
| Target-bound regional test | Both bbox predictions contradicted; next action changed to LEFT | Negative evidence can alter an action, while the two-competing-regions prerequisite still failed |
| Heterogeneous screen | 1/4 passed; shape/count failures included a lexical host guard rejecting the action word UP; color predictions were identical | Some failure belonged to the validator, some to the model; no broad non-spatial competence verdict |
| Structural Kaggle run | Four recorded tests, two metric matches and two none-of-predictions; ten actions, zero levels, eventual timeout | The new test interface was reached. A stale detector's zero count measured another gate |
| Sparse reasoning repair | Deep/summarizer/next-ordinary request switches checked locally as on/off/off | Per-call isolation exists; repaired target-GPU performance remains unmeasured |

In the globally reasoning-on structural run, 44 recorded calls contained 354,208 completion tokens, of which 322,305 were reported reasoning tokens. The problem was not absence of self-tests: it was extremely expensive decision formation under that configuration, plus the gap between a narrow metric and a broad claim. An any-change match did not establish the asserted directional movement. Procedure-saving APIs also exist, but these screens did not establish retained-procedure reuse or amortized inference. [Discovery appendix](appendices/discovery.md).

### Serving experiments already selected some things and rejected others

| Comparison | Measured finding | Retained decision |
| --- | --- | --- |
| Tuned Triton vs baseline, tiny fixed input | About +3.5%, below the improvement bar | Do not promote tuning from this result |
| Shallow MTP, tiny fixed input | About +58.6% steady decode | Earned a relevant workload test, not integration by itself |
| Repaired MTP, long Actor fixed input | About −29.7% steady decode; both this and the tiny fixed controls were greedy | Keep the working sampling repair, park MTP; do not explain the fixed-control reversal by stochastic sampling |
| Gregg 95k finite set, C1→C4 | Set completion faster by about 22%, median ready-to-completion slower by about 22% | Throughput and typical completion time disagree; no independent-game deployment claim |
| Chunk 2048→4096 ABBA | Gains missed 10% bar; graph settings and available recurrent checkpoints changed | Retain 2048; do not call larger chunks untested or isolate a kernel effect |
| Flash-Next | Eight v3 responses, but OOM violated the card; fixtures omitted the reported live context tail | PASS withdrawn; no valid matched speed or gameplay gain |

The earlier `decide()` semaphore test did not establish global HTTP serialization. A real packaged-worker loopback counterexample showed successor-triggered Actor work overlapping another game's request. This permits neither a blanket “parallelism is impossible” claim nor a claim that safe, faster multi-game operation is qualified. [Inference appendix](appendices/inference.md).

## 3. The ideas we keep proposing already have histories

| Recurring suggestion | Existing work and actual lesson | Disposition for future discussion |
| --- | --- | --- |
| “Give it compact, accurate state” | August candidate-local layout reduced tokens with unchanged supplied decisions; grounded position changed recovery; September successors implement decision packets | Inspect the exact missing/misused field in the current request before rebuilding the packet |
| “Make it discover the goal” | Typed goals, objective review/update closure, q4 and designer-player experiments already exist | Distinguish proposal, discriminator, execution, update, and measured benefit; a goal record is not discovery |
| “Add a planner” | Native planner paths and supplied-route ceilings already work under prerequisites; other packages expose different tools | Identify the missing prerequisite or access boundary, not a generic absence of planning code |
| “Check predictions after actions” | q6 feedback exists; later packages implement per-action interruption | Older post-batch failure cannot justify reimplementing the already repaired successor; verify exact deployment |
| “Use memory and retain learning” | Candidate retention, canonical publication, notes, summaries, and procedure interfaces have separate histories | Preserve useful work, but do not claim cross-run learning merely from storage or retrieval |
| “Show an executable demonstration” | Narrated demonstrations failed to test literal imitation; subsequent procedural screens exist | Read the later transfer/reliability results before claiming the test is still only proposed |
| “Force quicker commitment” | Model-visible budget telemetry degraded retained accuracy 8/16→0/16 in the matched Builder replay | Rejected as a default performance treatment in that family; any new version must name what differs |
| “Add more reasoning” | Several reasoning-enabled conditions consumed output/time budgets without a usable action; some clarified the failure | Reasoning presence is observable, but neither a sufficient explanation nor an automatic improvement |
| “Switch to CodeAct” | Native-vs-CodeAct, Actor-vs-Builder, and packaged access are distinct histories | Persistent computation can reduce crossings; merely having it does not establish uptake or better decisions |
| “Add contingent planning” | Bounded controller locally removes a model call under supplied plans; live model proposed none | Implementation exists; live adoption and benefit remain unresolved |
| “Use a faster model / MTP” | MTP depended on actual sampling/runtime compatibility; Flash-Next PASS was withdrawn after OOM and coverage review | Require relevant compatibility evidence; do not treat a synthetic speed result as Solver improvement |
| “Build an evaluation corpus” | Existing failure atlas, prediction-check corpus, parity suites, and offline experiment checks already exist | Name the population or inference the existing corpus cannot support; do not call it absent |

The individual appendices record which version each statement belongs to. This table is an anti-reinvention index, not an instruction to install every historical feature into the current candidate.

## 4. What failed, what was invalid, and what remains unexposed

These categories determine whether revisiting something is repetition or a meaningful new test.

| Classification | Example | What we are allowed to learn |
| --- | --- | --- |
| Supported narrow improvement | Typed serialization repaired supplied-option decisions; retained candidate survives rollback | Keep that exact gain within its tested scope |
| Valid negative result | Budget pressure worsened a candidate; grounded policy still failed routing | Reject the tested claim, preserving the conditions and residual uncertainty |
| Operationally invalid | OOM, provider failure, arm flag dropped, missing raw payload, request lifecycle failure | Diagnose the operation; preserve observations without promoting the planned capability verdict |
| Unexposed mechanism | No contingent proposal; no goal-boundary entry | Availability/adoption is the unresolved boundary; the mechanism's conditional quality was not tested |
| Real outcome without causal attribution | A level completed in the arm that never used its optional controller | Count the level; withhold the controller-benefit claim |
| Local mechanism evidence | Two actions from one supplied contingent plan using one model call | Executable integration works under supplied conditions; natural generation/general benefit unknown |
| Corrected assessment | q6 stale identity validator; prefix-cache assessor/glob and teardown classification repairs | Read the additive reassessment and original evidence together; do not erase the original verdict |

The proper denominator is part of every claim. A success-selected atlas cannot establish population failure frequency. Two arms with different prediction permissions do not identify the benefit of batching. A convenience sample below the context ceiling cannot establish long-context feasibility. A scalar predictor agreement can be vacuous if the encoder throws away the state distinctions that matter. [Early corpus](appendices/early-history.md), [performance](appendices/inference.md), [controller audit](appendices/controller.md).

## 5. Corrections to our recent conversation

| Previous statement | Corrected understanding |
| --- | --- |
| “Summary-boundary was locally qualified but not launched” | A live run is preserved, including 100 actions and two summaries; the README's old launch statement is stale |
| “The solver has shown no level-solving ability” | Several distinct histories contain completions; the recent controller `on` arm completed a level even though its optional mechanism was unused |
| “The body works; the intelligence is inadequate” | Both model behavior and substrate/interface failures are documented; the cause cannot be collapsed into an intrinsic model limit |
| “The next procedural test is still only a handoff” | Later procedural-transfer and self-test artifacts must supersede that earlier memory snapshot |
| “The next work is to add a packet, corpus, and prediction checks” | All have prior implementations or studies; the next question must name a missing, failed, or unexposed boundary |
| “Flash-Next is about five times faster / needs exactly 20% less context” | No valid matched Solver comparison establishes the speed claim; the observed context-tail defect is real, but exact cross-model token/template parity and a sufficient reduction are unresolved |
| “More actions per crossing show useful work” | Prediction eligibility and generated-helper divergence confounded the recent comparison; atomic decision-time opportunities were not recorded |
| “Structural self-tests were unreachable” | Four tests and four results exist in the Kaggle ledgers; the zero counter measured an inherited detector |
| “The offline experiment checks are still the initial 28-test version” | During this report, another review completed and the source now reports 49 passing tests and corrected issued-response accounting; this report did not rerun that suite |

Memory was useful for locating evidence but was stale about later launch status and procedural experiments. Current reports and raw observations take precedence. The report leaves memory files untouched.

## 6. What we can retain, and what we still do not know

### Retain as established knowledge

- We can run persistent model-driven game interaction and sometimes complete levels.
- Source identity, actual tool exposure, authoritative successor delivery, and lifecycle behavior materially affect whether a trial measures the intended Solver.
- More explicit evidence can repair specific observation/serialization failures; sufficient grounding does not automatically yield good routing or objectives.
- Model calls dominate the measured q6 time, while long context and serving policy change request cost.
- Mechanism availability, model adoption, conditional correctness, and gameplay improvement are separate achievements.
- Retaining a verified partial gain is useful even when publication is withheld; pressure to act can destroy gains if acceptance is weak.
- There is already substantial research infrastructure and decision history. Failure to consult it is a coordination defect, not evidence that it needs to be invented.

### Keep unresolved

- Which exact current candidate has repeatably better game outcomes under the available time/action budget?
- At which earliest preventable boundary do valid failures concentrate across a representative set of games?
- Do procedural demonstrations/self-tests transfer reliably beyond their supplied scenarios and remain useful in complete gameplay?
- Can the Actor naturally construct useful contingent plans, and do those plans improve outcomes after adoption?
- Can Flash-Next satisfy the actual workload's context, memory, output, and cleanup conditions, then improve equal-budget gameplay?
- Which historical repairs are actually inherited by the candidate selected for the next run, rather than merely present elsewhere in the repository?

These are questions, not authorization to run more experiments or a new architecture backlog. The report's purpose is to make the next choice start from preserved results instead of an erased history.

## 7. How to use this report without turning it into another project

Before reopening an idea, find its row and source appendix. Name the exact candidate, the unresolved exposure or outcome, and what differs from the previous attempt. If nothing differs, reuse the result. If an earlier result was invalid, name the invalidating condition. If it was negative, name why the proposed change could reverse it. If it succeeded, preserve the acquired behavior instead of rebuilding it under a new name.

This is a reading and decision discipline, not a proposal for another service, registry, agent, or evaluator.

## Verification and closing evidence boundary

The coordinator's [offline verification](verification-final-v2.json) checks report links, cited source identities, and selected raw outcomes: q6 100 actions/153 Actor calls/two summaries; controller off/on 20 actions each and zero/one level; and the supplied-route ceiling's 13 actions/one level. It does not certify every causal interpretation or rerun historical test suites. An [independent synthesis review](REVIEW.md) found no material overclaim within its stated scope. Markdown whitespace and code-fence checks passed.

Three source reports changed while the subagents were working. The coordinator read the new banners and [review disposition](../deployment-probes/20260909-experiment-checks/REVIEW.md): the changes strengthen the OOM withdrawal, reject a claimed causal JSON-prompt explanation, and correct v2 to three issued/preserved responses rather than two successful result-index rows. The review now reports 49 passing tests. This is additional offline audit progress, not new gameplay evidence. The initial inventory remains intact; [reviewed source updates](sources/reviewed-source-updates.json) records accepted replacement hashes and archived text. The first [verification attempt](verification-v1.json) is retained to show the detected drift.

Some older raw archives are missing, and not every discovered document was adjudicated. The appendices state those limits at each affected family. The report is a comprehensive reconciliation of the accessible research families, not a claim to have recovered all historical bytes or established current competition readiness.
