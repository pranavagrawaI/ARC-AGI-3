# Procedural transfer screen rerun 1 result

Verdict: **ERROR — zero paid/provider calls; no Qwen-quality evidence.**

The local relay preserved and rejected every received request because the rendered prompt lacked both the procedural diagnostic marker and the tagged discovery demonstration. It forwarded no request: `calls=0`, `reserved=0.0`, zero actions, experiments, model records, transitions, helpers, and procedures.

The preserved payload establishes the cause. `shared_local_solver.worker_environment()` did not include `ARC_PROCEDURAL_DIAGNOSTIC`, so the subprocess received the ordinary discovery configuration. Its prompt contained the ordinary `<arc_skill>` and `<decision_workflow>`, not the stripped diagnostic context. The client retried the local HTTP 500, producing 36 received copies representing three unique payload states; none crossed the provider boundary.

The card, `STARTED.json`, `RESULTS.json`, all received requests, and worker logs are retained. The result is not evidence for or against procedural transfer. Per the card, no relaunch occurred.

The candidate repair adds the diagnostic flag to the explicit worker environment allowlist and a regression test at that boundary. A future attempt requires another card and fresh output directory.
