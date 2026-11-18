# Constructor repairs verified; Kaggle launch blocked

Status: **BLOCKED_BEFORE_LAUNCH — no new Kaggle version, GPU allocation, provider request, or construction-policy outcome.** The explicit containment stop condition was applied. No retry, competition submission, live solver deployment, or git commit occurred.

## Verified local repairs

Fresh candidate derived from the immutable v4 constructor; original v4 frozen-source hashes reconcile in `LOCAL-VERIFICATION.json`.

- Preserve provider response bodies/transcript losslessly; project assistant tool calls into reviewed outbound request fields, excluding response-only `index`.
- Checkpoint the constructor before admission, after receiving a response, after actual usage accounting, and with tool delivery. Retain the shared result object on exceptions.
- Count actual transport starts separately from local validation/artifact work; reconcile completed receipts rather than calling a later local failure an unknown provider attempt. Preserve admitted completion reservations and reconstruct completed transcript pairs from durable raw artifacts.
- Treat missing usage as an explicit unknown-total abort. Audit that state without crashing; retain valid partial usage separately.
- Block the command-line execution path while Kaggle isolation remains unqualified. The inherited OpenRouter-shaped request contract is retained **only for offline v4 regression replay**, not claimed to be a vLLM contract. The transport endpoint no longer points to OpenRouter. This directory is not a deployable Kaggle package.

`python3 analysis/model-construction-kaggle-v1-20260913/verify_local.py` returned **44 tests passed**, exit 0, launch guard passed, all v4 frozen hashes unchanged. See `LOCAL-TESTS.log` and `LOCAL-VERIFICATION.json`.

Full-harness tests exercise the captured real v4 response → actual isolated Python inspection → second serialized HTTP request; successful fixture submission → freeze → one suffix assessment; first-request HTTP failure; first and second local pre-send failure; missing usage; exact 16,384 completion-token accounting with reserved commit; artifact failure; and audit tamper detection. Mock HTTP is used throughout. The successful witness source is a host-only machinery fixture, never constructor evidence or a model-generated outcome. The full local isolation suite also verifies absent evidence/verifier access, forged output rejection, branch conjunction/ancestry, bounded output, and forecast-before-assessment semantics.

## Containment blocker (primary historical outputs retrieved read-only)

1. `prior-containment/bwrap/result.json`, from version 1 of https://www.kaggle.com/code/pranavagrawai/prototype-tycho-bubblewrap-containment-probe : `bwrap` absent; direct user-namespace creation returned exit 1 / `Operation not permitted`.
2. `prior-containment/landlock/result.json`, from version 1 of https://www.kaggle.com/code/pranavagrawai/prototype-tycho-landlock-seccomp-containment-probe : Landlock ruleset query returned `ENOSYS`. Later failure entries cascade from that first failure and do not independently establish failed seccomp or resource-limit support.
3. Existing `solvers/tycho_nooa/src/tycho_nooa/_sterile_guards.py` installs Landlock before acknowledging readiness, so the sterile worker does not evade that prerequisite.
4. Existing `hardened_host.py` explicitly describes a diagnostic Unix-identity/seccomp boundary, not a general host sandbox. It does not create a filesystem allowlist/chroot. It has not been qualified here for the stronger constructor requirement that generated code can access only the current ABI arguments and no verifier/prefix/suffix files. Substituting it without qualification would weaken the guarantee.

Those primary outputs describe historical **CPU images**, not a newly measured RTX Pro 6000 GPU image. Therefore the conclusion is **no qualified supported remote containment route established**, not that Kaggle can never support one. No GPU was spent to rediscover a known prerequisite failure. Only allowlisted prior result JSON files were retained; automatically downloaded logs were removed before inspection.

## Serving discovery and remaining work

`deployment/kaggle/submission-20260912-v15/build.py` reuses runtime v11 and pins model dataset `jakobbrggen/qwen3-8-27b-fp8-hf-snapshot/versions/1`. Its routine request setup disables thinking using `chat_template_kwargs.enable_thinking=false` plus the existing runtime reasoning field. The prior package also enables one separate deep activation, which this proposed episode must not inherit. The dataset checkpoint is **not asserted equivalent** to the v4 OpenRouter alias.

A deployable construction-only notebook, actual vLLM contract qualification, final experiment card/manifest, immediate accelerator/quota preflight, one-shot launch, monitor, and new-run output audit were **not completed**, because containment qualification failed first. No final card was frozen and no launch identity consumed. The permitted next step is to qualify a supported remote boundary (including escape tests and startup fail-closed behavior) before spending on construction; no automatic relaunch is authorized.

## Interpretation

- Infrastructure: local constructor and audit machinery verified; Kaggle construction infrastructure unqualified.
- Construction completion / work item: not measured by a new model run.
- Historical suffix 6–8: exercised only by host fixtures; not new model support and not an untouched scientific holdout.
- General competence or performance improvement: not established.
