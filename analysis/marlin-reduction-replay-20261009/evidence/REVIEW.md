# Local candidate review

Candidate: only `analysis/marlin-reduction-replay-20261009`; no working native installation or frozen baseline was patched. Reviewed the complete five-file wheel patch, runtime hooks, independent W4 oracle, paired benchmark, notebook and raw audit against the handoff.

Corrections made before delivery:

- Capture graph copies are attached to the exact graph key and flushed after real armed target verification; a startup Python-hook snapshot cannot substitute.
- Override propagation was exercised through the actual GPTQ apply and Marlin runner functions, then through the actual fused function's two GEMM calls. Baseline defaults, target treatment, FP32 reduction and routing-weight placement are checked.
- Separate graphs share the same captured GPU tensors; allocation/timing observations preserve the full fused call and per-call reduction workspace.
- Scratch poisoning now fills the actual native GEMM scratch tensor inside a separate qualification graph. Clean graphs remain the performance contrast.
- Oracle rejects noninteger scales/types, any present zero points/biases/global scales/expert maps, and nonempty activation ordering. Empty zero-point tensors are rejected because presence selects a different scalar type in the caller.
- Numerical output arrays and all paired intervals are retained and audited; a test corrupts a raw timing interval and demonstrates that the derived advancement verdict is rejected.
- Large capture hashes stream in chunks, weight storage is checked before serialization, and weight artifacts are reused across selected replay points.
- Benchmark imports the preserved serving argument context and verifies installed candidate bytes before running.

Evidence:21 CPU tests pass. Tests include original packaged function execution with tensor/kernel doubles, capture serialization, known packed-layout golden values, numerical rejection, actual native scratch poisoning dispatch, notebook scope, raw timing audit and the original v3 raw warmup/profile audit. All30 original frozen v3 files, exact wheel hash and13 copied wheel member hashes verify unchanged.

Residual limits: no torch/CUDA runtime is available in the local test interpreter. GPU kernel calls, graph copy fidelity, numerical equivalence, timing, scratch memory and full model graph residency remain unqualified. Statement coverage/CC/CRAP is recorded under the prior stdlib AST contract, with no threshold and no quality-improvement claim; GPU orchestration and complete new-run raw audit retain substantial uncovered paths. The raw v3 audit validates retained baseline observations, not the new candidate. No GPU launch, quota polling, dataset upload, serving contrast, submission or promotion occurred.

No supported local defect remains in the reviewed scope. The next gate is one separately authorized, one-attempt GPU qualification under EXPERIMENT-CARD.md; failure remains evidence and cannot silently trigger a retry.
