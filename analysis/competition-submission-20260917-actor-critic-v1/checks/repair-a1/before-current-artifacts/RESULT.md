# T3 result: locally qualified; no launch authorization

Final Python 3.12 qualification passed: **230 pytest tests, 0 failures**, three end-to-end notebook/gateway scenarios, and three offline installed-SDK checks. The 110-game scenario recorded **550 Actor calls and 110 critic calls**, iterated every supplied synthetic game, and produced a complete 110-game gateway-fixture parquet. The reserve scenario recorded 2 Actor/2 critic calls across two games after limiting Actor to one slot each. The critic-ceiling scenario recorded 24 Actor/8 critic calls across two games, stopping at 12/4 per game. Workers were reaped; request and server-count reconciliation passed and perturbed counts were rejected.

Identity: `pranavagrawai/arc3-ac-competition-20260917-v1` (title/SDK slug equal).

- Manifest SHA-256: `5cc3f11404566a29a0ccbc1b6fdcccca90c0154663e3e4eed5b0935a366fc70e`
- Card SHA-256: `eb8ea21645e534e7289e480b2b262179598545c78ef0f5a74de6c240a4a8d1af`
- Notebook SHA-256: `910ac026855425732d7f9c8e5eabfdccd1607ff43d186d34c420902c4fe92183`
- Recomputed source bindings: 84; donor bindings: 65.
- Gate: expected **HOLD / exit 2**, because independent review and authorization do not exist. No real consumption marker exists.

`checks/final-checks.json` contains exact counts, artifact hashes and evidence paths. `checks/commands.json` records reproducible commands. `REVIEW-REQUEST.md` is the exact T4 review request. No self-approval was created.

Validation scope: exact embedded assets were materialized from the generated notebook, and the official Agent.main and official REST API executed with scripted model replies. Packaged import origins were inside the disposable framework; dependency imports were site-packages, with all 32 pinned tycho_nooa source hashes checked. The final manifest reconciles after testing, with no package bytecode or source drift. All donor hashes reconcile and tracked Git status matches the starting snapshot. All authored files are under the two authorized new directories.

Earlier evidence is retained: initial inherited tests had 4 path-resolution failures, the first gate/payload run had 3 deterministic-ZIP mismatches, and the first inherited entrypoint run had 2 failures (an obsolete budget expectation and an in-flight development freeze change). The first three two-game fixture attempts failed on fixture setup or an overstrict request-count assertion. The initial successful Python 3.13 suites and 110-game fixture are superseded for no-shadowing qualification because tycho_nooa came from an editable checkout. The final Python 3.12 results above resolve that boundary; no failures are hidden or counted as final passes.

Material limits (7 unexercised remote categories):
- GPU/container offline bootstrap and model warmup on RTX PRO 6000
- real model latency and gameplay capability
- private Kaggle gateway parquet writer
- actual private 110-game inventory
- provider competition timeout/queue-overhead semantics
- live one-push and competition submission
- fresh quota and remote exact-version retrieval

The local parquet writer is an explicit callback implementing the published output contract, not Kaggle's private sidecar writer. Supervisor reconciliation uses observed fixture model requests plus a simulated one-call warmup count; it does not claim a GPU SGLang warmup occurred. The 900-second provider uncertainty reserve and 15-second tool margin are policy allocations, not measured remote bounds. The full pinned GPU wheel closure was not installed locally. These results establish local integration and bounds, not model capability, remote validity or scientific improvement. The T2 remote-public qualification prerequisite remains for T4 to resolve before any submission.

No GPU, paid model call, external service write, kernel push or competition submission occurred. Read-only package downloads installed parquet dependencies only into the new analysis directory; existing Python environments were not modified.
