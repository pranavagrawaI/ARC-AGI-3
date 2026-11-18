# Ready: captured Actor baseline versus shallow MTP

Upload actor_mtp.ipynb with the sources in kernel-metadata.json. Attach
arc-prize-2026-arc-agi-3, select RTX PRO 6000, disable internet, and cap the
session at 7200 seconds. This preparation does not launch it.

All 14 actual request bodies are sealed, including attempts 4, 12 and 13 whose
recorded answers called undeclared tools. Each new answer is checked and retained;
invalid output does not prevent the other arm running. No tool code is executed.
HTTP/server/lifecycle failures still stop the trial.

Requests preserve sampling and output-limit omissions. The pinned engine uses
--sampling-defaults model; mounted generation_config.json must equal the capture.
No silent 4096-token limit is added. Runtime deadlines bound cost.

Each arm: excluded 64-token warmup, 14 cold-cache natural requests, two fixed
1024-output native diagnostics on the longest captured prompt (~21k). Diagnostics
measure decode without chat grammar. No forced-action, compression, thinking-on,
concurrency or closed-loop gameplay coverage. Both arms receive frozen historical
inputs, never the candidate's preceding outputs.

Rebuild and verify:

```bash
python3 seal_capture.py
python3 build.py --corpus corpus.json
python3 seal_capture.py --check
python3 build.py --corpus corpus.json --check
../.venv/bin/python test_local.py
```

Read EXPERIMENT-CARD.md. Review behavior, output counts, limit outcomes and latency
together: shorter or invalid answers are not automatically faster equivalent work.
The separate integrated notebook tests gameplay, not matched-request performance.
