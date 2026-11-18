# score-conversion-cpu-20261013

Question: Is the observed s5i5 coupling defect locally repairable, and does a validated simulator suffice for completion at the original search horizon?
Posture: Component diagnosis using a deliberately privileged reference simulator.
Contrast: Original archived predictor; one-rule S-grow descendant repair; archived successful chain predictor. Reference search horizons 27, 31, 44 from two stalled states.
Prediction: A one-rule repair improves the witnessed contradiction but leaves other errors; even a correct model fails at a horizon shorter than its solution.
Measure: results.json contains explicit selected/called/compared/matched/mismatched/rejected/abstained/error counts, exact counterexamples, bounded search receipts, native-engine completion receipts and score arithmetic.
Cap: Local CPU only; 180 wall seconds; 10000 distinct search states per search; six searches, at most 2 native recovery plans, at most 44 recovery actions per plan; zero network requests, model inference calls, GPU use or spend.
Stop: Finish the fixed matrix. Abort on source/state mismatch, a failed known historical-score check, elapsed cap, unexpected native divergence, or search state cap. Preserve the failed receipt; no automatic retry.
Invalid if: Native predecessor frames do not match archives; reference source changed; empty comparisons called successful; a game-over frame mistaken for level completion; later evidence represented as causally available to the original model.
Result map: Narrow repair generalizes -> candidate for repair elicitation; fixes only known cases -> require prospective predictions, no promotion; reference cannot solve within adequate bounds -> investigate model/planner; reference solves only with longer bounds -> separate simulator repair and search scope in next tests; native mismatch -> reject model-only claims and diagnose. All outcomes preserve incumbent; none establishes feedback efficacy.
Outputs: analysis/score-conversion-20261013/results.json and native-receipts.json. Script records source hashes. Retrospective transfer checks are diagnostic, not a statistically independent test set. Human-authored narrow repair is not an LLM treatment.
