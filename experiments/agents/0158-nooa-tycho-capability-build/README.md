# 0158 — NOOA/Tycho Grounding Capability Build

0158 is a bounded capability build for prospective visual binding before one public ARC action. It implements only immutable empirical evidence, model-authored grounding claims, contrastive action-effect commitments, exact non-oracular two-axis verdicts, G0–G2 qualification, and human-reviewed failure packets.

The build deliberately stops at G2. It contains no world model, transition model, outcome model, planner, builder, retrieval memory, helper persistence, or multi-action submission surface. Reaching an internal gate is not gameplay success; gameplay success remains `levels_completed >= 1` in a separately approved fresh LS20 run.

```text
CANDIDATE: 0158-nooa-tycho-capability-build
PARENT: 0002-NOOA split-runtime substrate plus Tycho evidence conventions
CLAIM CLASS: Capability Build
MUTABLE MECHANISM FAMILY: grounding claim, reciprocal prospective action-effect contrast, exact verifier, G0-G2 gates
FIXED INVARIANTS: Gemma authors semantics and actions; one action; non-oracular host; G3+ absent
MECHANISM MEASURE: three linked claim-anchored informative effects
BEHAVIOR MEASURE: separately approved fresh LS20 level completion
TASK MEASURE: levels_completed >= 1
EVIDENCE TIER: DEVELOPMENT
```

## Document map

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — runtime and capability boundaries
- [`CONTRACT.md`](CONTRACT.md) — frozen public LS20 contract
- [`QUALIFICATION.md`](QUALIFICATION.md) — offline verification commands and claims
- [`PROVENANCE.md`](PROVENANCE.md) — pinned sources and exact adaptations
- [`REPORT.md`](REPORT.md) — implementation report
- [`docs/`](docs/README.md) — indexed design decisions and live-run reports

## Runtime boundary

The protected host runs in the repository's Python 3.14 environment. Each model artifact launches the pinned NOOA dependency in a Python 3.12 process and a fresh CodeAct sandbox. Only serialized immutable evidence crosses IPC. 0158 uses NOOA for agentic-method/runtime structure, **not** for unrestricted live-object pass-by-reference.

## Qualification

No live provider or ARC environment is contacted by the qualification suite. See `QUALIFICATION.md` for commands and evidence boundaries.

## First live contact

The separately approved 2026-08-11 LS20 run stopped before any ARC action on an uncertain provider 429. It is a terminal substrate result, not a Gemma capability result. See `docs/runs/LIVE-RUN-20260811.md` and the preserved run directory; the consumed approval cannot reopen that trajectory.

A separately approved retry reached the model but again stopped before any ARC action: Gemma exhausted the 4,096-token generation cap while narrating the first grid and returned no claim. See `docs/runs/LIVE-RUN-RETRY-20260811.md`. The retry scorecard and approval are also closed.

The next approved run doubled output capacity to 8,192. The length failure disappeared, but Gemma exhausted six CodeAct iterations without returning a claim; therefore the conditional 16,384-token run was not opened. See `docs/runs/LIVE-RUN-8K-20260811.md`.

The following approved run raised the CodeAct ceiling to 12 while retaining 8,192 output tokens and the 44-action cap. Gemma completed a claim artifact after six calls, but the host rejected its numeric `rows` as an invalid binary cell-mask encoding before any action. Because the visible schema did not define the required `#`/`.` representation, the evidence does not yet separate model protocol failure from interface underspecification. See `docs/runs/LIVE-RUN-12ITER-20260811.md`.

The between-run correction keeps binary masks but makes their encoding explicit in Gemma's phase instruction and structured schema. The exact failed numeric artifact is now an offline regression case; no scored run was resumed.

The first approved run of that corrected candidate encountered HTTP 429 on provider call four after three successful calls. It closed with zero actions as a substrate failure, leaving the correction behaviorally untested. See `docs/runs/LIVE-RUN-MASK-SCHEMA-20260811.md`.

A fresh retry on 2026-08-12 reproduced the same boundary: three successful calls followed by HTTP 429 on call four, again with zero actions. See `docs/runs/LIVE-RUN-MASK-SCHEMA-RETRY-20260812.md`.

A second retry failed with the same HTTP 429 class and hashes on provider call one. After three consecutive corrected-candidate transport stops, further identical live retries are blocked pending a transport change. See `docs/runs/LIVE-RUN-MASK-SCHEMA-RETRY2-20260812.md`.

The next-run candidate now pins `coreweave/bf16` exclusively, with fallback disabled. This changes provider and quantization; it does not reopen or reinterpret prior runs. See `docs/decisions/PROVIDER-SWITCH-20260812.md`.

The first CoreWeave run cleared transport with five successful calls. Gemma used the corrected binary-mask encoding but duplicated its primary mask as the sole alternative, which the host rejected before any action. Because distinctness was not model-visible, the result remains inconclusive between model protocol failure and interface underspecification. See `docs/runs/LIVE-RUN-COREWEAVE-20260812.md`.

The between-run representation correction retains binary membership masks, makes every claim invariant visible and repairable inside CodeAct, and permits `alternatives: []` because alternatives do not currently participate in verification or promotion. See `docs/decisions/REPRESENTATION-DECISION-20260812.md`.

The first run of the complete claim contract stopped on CoreWeave HTTP 429 at provider call two, after one successful call and before any claim or ARC action. It is transport evidence only. See `docs/runs/LIVE-RUN-COMPLETE-CLAIM-20260812.md`.

The next-run transport policy retries HTTP 429 up to five times with audited exponential backoff or bounded `Retry-After`, while retaining the single pinned provider and disabled fallback. See `docs/decisions/TRANSPORT-RETRY-DECISION-20260812.md`.

The bounded-backoff run encountered no 429: all six CoreWeave crossings succeeded. Gemma adopted empty alternatives and binary masks but produced seven-character rows for an eight-column bbox. Pydantic rejected it, and `max_retries=1` left no in-artifact repair opportunity. See `docs/runs/LIVE-RUN-429-BACKOFF-20260812.md`.

The next-run candidate raises the bounded CodeAct error budget to four while retaining 12 total iterations. Exact schema feedback can now lead to an in-artifact correction before action authority. See `docs/decisions/CODEACT-REPAIR-DECISION-20260812.md`.

The four-error live run exercised repair feedback but Gemma repeated an empty `return_result` call four times despite exact missing-field guidance. All 12 provider crossings succeeded; no ARC action executed. See `docs/runs/LIVE-RUN-CODEACT-REPAIR-20260812.md`.

The next-run candidate exposes only `execute_python`; completion occurs inline with `return_result(candidate)`. A model-visible `make_mask()` derives dimensions and validates membership syntax without choosing semantics. See `docs/decisions/INLINE-RESULT-DECISION-20260812.md`.

The inline-result live run reached CoreWeave without a 429, but Gemma used the full 8,192-token output allowance reproducing the large frame in reasoning and made no tool call. It stopped with zero ARC actions. The immutable generic substrate label is superseded by a separate evidence-backed `model_generation / model_output_exhaustion` correction. See `docs/runs/LIVE-RUN-INLINE-RESULT-20260812.md`.

The attempted required-tool run made five `execute_python` calls but did not actually carry `required` on the HTTP wire: pinned NOOA overwrote the wrapper value with `auto`. The treatment is invalid, and the run stopped at zero actions after repeated wrong CellMask submissions exhausted the repair budget. See `docs/runs/LIVE-RUN-REQUIRED-TOOL-20260812.md`.

Qwen3.6-27B passed synthetic provider qualification and produced 0158's first accepted live grounding claim, but no action executed. Its commitment used a reasonable invented bounded-replace dictionary that the permissive sandbox accepted and the hidden stricter host parser rejected. The corrected result is interface underspecification, not Qwen semantic failure. See `docs/runs/LIVE-RUN-QWEN36-MODEL-SWITCH-20260812.md`.

The exact effect union and constructors passed two-phase qualification, but the live run stopped before action because claim submission exposed an evidence-set digest beside record IDs and required manual current-frame citation. The post-run interface now binds the current observation mechanically and removes the digest from the model task. See `docs/runs/LIVE-RUN-QWEN36-EXACT-INTERFACE-20260812.md`.

The semantic-investigator interface then passed the offline suite and exact two-phase provider qualification. In the fresh live run, Qwen produced an accepted claim and anchored commitment and the host issued ARC action 1—the first action from 0158. The successor contradicted Qwen's predicted replacement (`PRIMARY_REPLACEMENT_MISMATCH`), so the non-resumable run stopped with one action and zero levels completed. See `docs/runs/LIVE-RUN-QWEN36-SEMANTIC-INTERFACE-20260812.md`.

The autonomous 44-action condition was exercised in three fresh, non-resumable attempts. The strongest run recorded six successors: two contradicted predictions followed by four supported exact replacements, including one complete G2 cycle, before an ARC service timeout made action 7 uncertain. Later transport hardening removed that 10-second boundary, but the next fresh run stopped after action 1 when Qwen exhausted all 12 commitment iterations rediscovering already-visible constructors. The cap was not reached and no level completed. See `docs/runs/LIVE-RUN-QWEN36-AUTONOMOUS-44-20260812.md`.

The next candidate isolates that procedural closure failure. Commitment now uses one atomic helper call, and code physically permits at most three successful exploratory cells before requiring a direct submission-only cell. The old two-step constructors and `submit_probe` are absent. Semantic handoff remains unchanged so the next run tests one mechanism. See `docs/decisions/ATOMIC-COMMITMENT-DECISION-20260812.md`.

That candidate passed offline and exact hosted-worker qualification, then recorded eight confirmed LS20 successors with all eight commitments closing atomically. It surpassed the prior six-confirmed-action maximum and reached one G2 cycle, but did not complete a level. The ninth claim exhausted 12 iterations because the hard gate required one AST expression while Qwen repeatedly used a constructor assignment followed by submission. The next evidence-selected change is constructor-only submission grammar, not more iterations or semantic handoff. See `docs/runs/LIVE-RUN-QWEN36-ATOMIC-44-V21-20260812.md`.

The host gate now distinguishes deterministic artifact construction from semantic exploration. After three explorations it accepts only phase constructors and literals assigned to local names followed by the atomic submitter; evidence reads, imports, control flow, and arbitrary computation remain physically rejected. The exact Qwen-shaped claim and a paired forbidden-read case are qualified through the real NOOA sandbox. No new live run was made for this host-only correction. See `docs/decisions/CONSTRUCTOR-SUBMISSION-GRAMMAR-DECISION-20260812.md`.

The fresh v22 run then reached all 44 actions, confirming the constructor grammar fixed v21's claim-closure failure. Its precommitted same-action recurrence measure passed, but the claimed action-semantic interpretation did not: Qwen predicted an automatic next-column progression that occurred under actions 1–4 and explicitly inferred that the chosen action did not matter. The run demonstrates prospective history use, not distinct opaque action semantics, and completed no level. See `docs/runs/LIVE-RUN-QWEN36-ACTION-SEMANTICS-V22-20260812.md`.

The next candidate requires reciprocal prospective predictions for the same claimed region under distinct legal actions. A locally accurate effect remains action-specificity-inconclusive until two selected actions each succeed and their earlier commitments predicted one another's effects. Position-invariant replacement signatures prevent successive locations of one automatic progression from qualifying as different action effects. No live result is claimed yet. See `docs/decisions/CONTRASTIVE-ACTION-MAP-DECISION-20260813.md`.

The fresh v23 run stopped before any ARC action. Qwen reached the new replacement constructor but passed one `cell_after(...)` dictionary where the undisclosed helper signature required a list, then exhausted the commitment artifact. A one-factor offline replay accepted `[cell_after(...)]` and rejected the original form. This is interface-legibility evidence, not a test of reciprocal action learning; the run remains closed with Pranav's causal review pending. See `docs/runs/LIVE-RUN-QWEN36-CONTRASTIVE-ACTION-MAP-V23-20260813.md`.

The next candidate removes that bookkeeping trap rather than merely documenting it. The claim phase alone retains `make_mask()`. The commitment phase now exposes `cell_after(d_row=..., d_col=..., value=...)` relative to the accepted primary mask and variadic `predict_cell_values(cell_after(...), ...)`; it physically omits `make_mask()` and `predict_replace()`. The host expands relative assertions into the same exact absolute `Replace` protocol record, so verifier semantics do not change. No fresh live run is authorized or claimed. See `docs/decisions/PRIMARY-RELATIVE-CELL-API-DECISION-20260813.md`.

The v24 live run confirmed that API closure: three relative-cell commitments were accepted and each released a confirmed successor without schema repair. All three semantic predictions were contradicted because Qwen continued to claim the static 3x3 marker instead of the moving 5x5 block. Actions `1, 1, 2` visibly moved that block up, up, then down by five rows, but Qwen never made those translations load-bearing. The fourth claim artifact then stopped on a fatal-classified HTTP 408. The run is terminal with zero levels completed. See `docs/runs/LIVE-RUN-QWEN36-PRIMARY-RELATIVE-CELLS-V24-20260813.md`.

The next candidate separates mechanical evidence preservation from semantic judgment. After each probe, the host reports only changed-cell counts inside and outside the prior primary mask; the model must bind an explicit retain/revise/abandon update and account for any residual as explained, auxiliary, or unresolved. A general evidence-conservation instruction prohibits silently improving fit by narrowing the explanandum, while leaving relevance, identity, and causal meaning entirely model-authored. See `docs/decisions/EVIDENCE-CONSERVATION-DECISION-20260813.md`.

The v25 run executed six confirmed actions and showed the intended first-order correction: after `0` inside and `52` outside changes contradicted the 3x3 proxy, Qwen explicitly abandoned it and moved onto the action-varying block. It still modeled only a predictable 2x5 slice, semantically dismissing 42 outside changes without integrating the full translation. Claim 7 then used `residual_status='none'` despite those changes; a late protected-host check terminalized the run. No level completed. See `docs/runs/LIVE-RUN-QWEN36-EVIDENCE-CONSERVATION-V25-20260813.md`.

The next candidate removes every residual-closure label. Visible change remains immutable evidence but is not a debt that the chosen-action model must absorb: independently evolving interface or environment state may change after every action. Each post-probe claim instead carries a full model-authored register of scoped provisional heuristics with evidence, operational attention (`active`, `watch`, or `defer`), and a reconsideration condition. Prior heuristic names and evidence cannot disappear, and no entry can become explained, confirmed, closed, or certain. See `docs/decisions/PROVISIONAL-HEURISTIC-REGISTER-DECISION-20260813.md`.

The v26 run reached all 44 actions with 44 confirmed successors and no substrate failure. The provisional register preserved rejected candidates without closure, and Qwen learned an active `shaft_motion` heuristic stating action 1 moves up five and action 2 down five, including an upper boundary. Yet it made zero `Translate` commitments, selected no actions 3 or 4, earned zero action-specific promotion, and never separated the bottom progression as a watched scoped heuristic; after action 43 it misattributed that progression to action 2. No level completed. See `docs/runs/LIVE-RUN-QWEN36-PROVISIONAL-HEURISTICS-V26-20260813.md`.

The matched DeepSeek V4 Flash, Qwen3.7 Flash, and Muse Glimmer probes all reached semantic errors but then terminated at the same procedural seam: the three-call evidence lockout plus full-register/evidence restatement. Because that seam prevents clean model/interface attribution, the next candidate was simplified through four delete-and-recreate passes. It now exposes one `inspect_evidence()` snapshot, keeps evidence readable within the bounded artifact iteration budget, accepts changed-only `heuristic_updates` while the host conserves the full register, and distinguishes fixed-coordinate `predict_cell_values()` from object `predict_translate()`. No model or ARC run has been made with this candidate. See `docs/decisions/DELETE-CENTRIC-INTERFACE-DECISION-20260814.md`.
