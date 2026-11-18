# gpu-q5 — failed before gameplay

Private version 1 ended with remote status ERROR, first recorded by the collector at 2026-10-03T23:01:01+00:00. The notebook log reached server readiness at 595.867 seconds and failed at 602.067 seconds, before Benchmark.run. No hybrid gameplay, construction trace or score result exists. This is an execution failure, not a capability zero.

Cause: hybrid/runner.py:76 recorded game_names with getattr(game, "env_name", game.label). Python evaluates game.label before getattr chooses env_name. Actual GameAPI objects expose env_name but no label, causing AttributeError. The CPU rehearsal used ExampleGame and the previous small-set test stub supplied label; both missed the production input shape.

The persistent collector completed its job: it preserved qualification-start.json, watchdog-receipt.json, serve.log and notebook.log, then reported missing benchmark/config/index in AUTOMATIC-AUDIT.json. Their absence is explained by the pre-game crash. FAILURE-AUDIT.json preserves the diagnosis. The service exited after collection; it is no longer running. No matched pair or retry launched.

The editable candidate now uses an explicit conditional so the fallback is evaluated only when needed. The small-set regression now uses actual GameAPI instances; an additional generated-runner test exercises offline ar25/ls20/sb26 with legal scripted actions. The next source bundle has a distinct source-v4 identity and has not been uploaded. Frozen q5 notebook, metadata, source and receipts remain unchanged.

Notebook execution lasted roughly ten minutes, but the campaign ledger conservatively charges the full one-hour cap because an exact allocation billing receipt is unavailable. Eight GPU-hours remain after q4 and q5. Any new GPU qualification needs a distinct frozen card after testing the packaged real-GameAPI path; the failed run must not be silently replayed.
