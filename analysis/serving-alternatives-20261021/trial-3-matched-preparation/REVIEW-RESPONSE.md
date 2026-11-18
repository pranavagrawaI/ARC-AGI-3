# Resumed review: protocol hold

The resumed Claude Opus 5.5 review completed after its initial API rate-limit failure. Raw result: review-resumed.json. It is a completed independent review, not a launch approval. No serving GPU has run.

## Decisive local observation

occupancy_sensitivity.py builds corpus gaps ONCE under the original simulator control, preserves alpha .523, then changes only a hypothetical decode-rate multiplier. PYTHONHASHSEED=0 and the original sim.py/corpus hashes are recorded in occupancy-sensitivity.json. Measurement is 600–1500s, matching this draft.

Control multiplier1.0 gives mean8.975 / P9 .739. Multiplier1.20 gives mean8.664 / P9 .606, failing both the absolute tail band and paired matching. Multiplier1.50 also fails. This is a counterexample to a neutral matched-occupancy gate: shorter service leaves a larger fixed-gap fraction. It is not measured candidate throughput, and the multiplier is not an aggregate U_resp speed ratio. The simulator's decode phase is also not a qualified native residence trace.

The resumed review's assertion that the 1.10 case necessarily fails does not reproduce under the fixed seed: our1.10 case passes narrowly. Set iteration/scheduling changes exact rows. The material1.20/1.50 counterexample remains. Do not generalize it to all engines or all faster implementations.

## Protocol choice for coordinator review, not adopted here

Preferred next local design: a separately identified saturated capacity screen, using the same full-image corpus and S10 but zero artificial tool gaps in BOTH arms. Retain U_resp as primary; per-running-stream rate remains descriptive. In the same simulator, alpha0 keeps paired mean/tail differences within the draft's matching tolerances through1.50x. However its absolute occupancy exceeds the old production bands: those bands cannot be reused. New bands require an explicit rationale and precommitment, native adapter qualification, and sensitivity to prefill/cache variation. This would support saturated serving capacity only, not production-traffic or gameplay adoption.

Alternative: retain representative fixed-gap replay, treat achieved occupancy as an outcome, and redesign the comparison without conditioning qualification on a speed-dependent matched mediator. That changes the scientific contrast and still needs workload/identifiability review. Scaling gaps from each engine's observed speed would change offered traffic and introduces adaptation; it is not a drop-in fair match.

Do not spend the proposed3h on the current draft as a default feasibility exercise. The probability of passing each native gate is unestimated: one historical control failure and a biased simulator do not support calibrated numerical probabilities. A longer warmup does not establish correction of that failure. A failed comparison remains feasibility-only, but that fallback alone does not justify this allocation.

## Reducer and lifecycle findings

- Original lifecycle repairs remain supported by the resumed reviewer: duplicate responses, request errors, boot/trace continuity, invalid timestamps, phantom streams and send ordering.
- Q3 malformed artifacts reproduced before repair in resumed-probes-before.json. usage=None, missing status/window, and a string legacy gauge raised exceptions. They now produce FEASIBILITY_ONLY; the regression also covers null arm/pair inputs. The current suite has20 test methods. This is a narrow repair, not approval of the protocol.
- Q1 remains OPEN: the reducer accepts a claimed warmup boolean and does not derive its fixed window from replay phase events or compute every initial holder's warmup completion. A supplied impossible replay_start_t still passes the old synthetic rate fixture. A live artifact reducer must derive these facts, not accept the flag.
- Q2 remains OPEN: the prototype accepts upstream mix/fidelity summaries. Before any live screen, compute prompt/image/cap right-closed histograms, early stops and completion fractions from measured request/response records joined to the immutable corpus; validate source/image/processor identities from their raw receipts. Unit fixtures that assert those summaries do not establish this path.
- Q4 remains OPEN in the retained draft: replay phase allowance exactly equals warmup+measurement. Reserve setup/cutoff/scrape slack explicitly. One bounded proposal is1560s per replay and840s final collection instead of1500/960, preserving the10200s supervisor cap; no revised execution budget is adopted yet.
- Q5 hook specification now explicitly includes native abort/finish/free paths. Adapters remain unimplemented; actual lifecycle coverage is unproved.
- Corpus semantics need explicit repair: finite individual games may finish normally (the simulator finishes3–6); only inability to maintain S10 because too few unfinished game streams remain is corpus starvation. The retained exhausted_games field is too ambiguous for a launchable reducer. Report finished-game counts separately and derive the starvation gate from admission events. Neither zero individual finishes nor an empty wait queue alone establishes the intended condition.

## Prompt reconciliation

prompt-count-reconciliation.json independently confirms a constant11-token recorded-minus-local offset for all1369 records. The local two-arm rendered strings and token counts agree with each other; that is weaker than runtime equivalence. Offset cause remains unresolved. Do not claim that the11 tokens prove a particular serialization, template or image-path mechanism. Historical maximum joint129998 still fits139264; runtime native prompt IDs/counts and image-grid receipts remain required. Fast/PIL pixel-processing differences are disclosed as a runtime confound.

## Current boundary

Card/config are DESIGN BLOCKED and unfrozen. Original pre-review sources are preserved in before-resumed-review/. No alternative traffic protocol, GPU cap or launch was approved. No stage/commit was performed. Trials1/2 stay unchanged. Next useful work is to settle the scientific contrast locally and repair its raw-evidence reducer before constructing or freezing a GPU packet.
