# Actor–critic construction: Kaggle packaging qualification

**Posture:** packaging and local verification only. Zero kernel pushes, dataset pushes, competition submissions, or remote-model requests authorized by this card. `launch_once.py` always refuses. The coordinator owns any subsequent exact-hash review and launch gate.

**Question:** Can the successful v2 automatic Actor/critic/construction integration execute on private Kaggle with locally served Qwen3.8-27B, without gateway credentials or fabricated provider receipts?

**Identity:** `pranavagrawai/actor-critic-construction-kaggle-r2`. Private notebook; internet disabled; competition attachment exactly `arc-prize-2026-arc-agi-3`. Dataset/image/accelerator fields inherit the pinned donor metadata. The runtime must independently observe one RTX PRO 6000 and verify the mounted model before serving it.

**Prior qualification:** successful OpenRouter manifest `6fb19b860a7bce1a104e5e2dc2f5f9210aca201d929ab545e5480018ef9e0661`; 422 protected source bindings must remain unchanged. Its success does not establish Kaggle runtime compatibility.

## Fixed bounds

- Server startup: 1200 seconds. Model admission: 900 seconds. The supervisor R8-L3B request timeout remains 180 seconds (its fixed runtime-mode contract), separate from the application per-POST cap of 30 seconds. One runtime-owned warmup stays inside the 24 total requests. This restores donor compatibility, not a global bounds increase.
- At most **24 total server chat requests**, including the donor's one runtime-owned warmup. The application relay therefore admits at most 23 Actor/summarizer/constructor/critic calls. Actor plus summarizer remains capped at 20. This is a conservative allocation within the original 24-call ceiling, not an extra warmup allowance.
- At most eight actually executed environment actions. Constructor: 45-second soft budget, 50-second hard process deadline, two requests; per-POST HTTP timeout 30 seconds.
- One-hour GPU reservation ceiling. Bootstrap work deadline remains 3100 seconds; the dedicated notebook-descendant guardian has a 3500-second deadline plus an eight-second forced retirement bound. The production model supervisor's application subprocess is bounded by 1020 seconds, including cleanup reserves. None of these local controls is provider cancellation.
- Before any authorized launch the coordinator must bind the supported provider run timeout, exact returned job/version, quota reservation and terminal-state collection. Queue/billing uncertainty remains a provider-level gap until that gate and final readback.

## Required path and evidence

Self-contained notebook materializes the source allowlist, then fresh isolated interpreters run GPU discovery and **actual sealed candidate/inspection qualification before installation, model-server startup or warmup**. The hashed donor offline installer/venv/model-supervisor machinery remains, but its old game entry, nested archive and constructor case are absent.

The application driver is `test_join_qualification.build_agent`, with its test Agent base and the real v11 worker/offline public game. It is **not `Agent.main`**. v2 selects the executed prefix automatically, previews a frozen prediction for the Actor's selected action, assimilates only its matching executed successor, and preserves the incumbent critic trigger. No semantic work item or candidate source is shipped to the live model.

Raw local HTTP bodies, logical and projected requests, usage, latency, constructor progress, host events, environment transitions and retirement evidence are retained. Exact local model path and actual server startup argv replace the OpenRouter identity audit. The supervisor rereads local receipt accounting and reconciles it against server access counts, including its warmup; no successful Solver sessions or gameplay outcomes are invented to satisfy its old schema.

**Success criterion:** frozen model constructed by the served model; completed same-digest predict/assimilation bound to an actual action/step; genuine critic call; environment/host action agreement; complete valid local receipts; bounded admission and confirmed cleanup. ABI replay/assimilation remains unscored. This is an integration criterion, not prediction accuracy or game-playing improvement.

**Stop:** reject next model admission immediately once the integration predicate is met; abort-retire/reap the worker without fictitious finish or another environment action. Missing usage, wrong identity, transport failure, failed seal, drift or unresolved cleanup is failure, not zero-cost success. Constructor timeout cannot publish a late freeze/result. The outer subreaper handles notebook death and independently sessioned descendants.

**Invalid if:** credentials, raw history, sealed traces, game-specific constructor case, fake CoreWeave/provider receipts, old fixture entry, source drift, wrong model/GPU, unsupported local wire, missing receipt, or false `Agent.main`/remote-success claims.

**Result map:** local failure → repair only owned package files; local pass → freeze for independent exact-hash review. Real GPU/model/installation/teardown/quota remain unexercised locally. Any remote failure after an eventual authorized launch requires a new identity; do not mutate launched bytes.
