# Execution pre-mortem

Failure scene: two GPU sessions are spent but the comparison is uninterpretable, or a lost response causes a duplicate launch.

- Duplicate/lost push response -> attempt marker created exclusively before API mutation; reserve the whole arm cap before push; never push that arm again. Test with a fake transport that accepts then raises.
- Unhealthy control or incomplete download -> collection stages without deleting failures; only a completed collection and runtime-health report permit treatment. Test healthy archived output, short runtime, wrong GPU, failed platform status and missing files.
- Concurrent owned GPU work or quota uncertainty -> fresh live active-session and used-plus-reserved check before each arm. Require no active sessions and zero reserved time. Test admission with active, unknown and insufficient quota.
- Silent config/source drift -> frozen local hashes and source readback compared by all code-cell source/type. Test changed source rejection.
- Infinite observation/transfer wait -> platform hard cap 6000 s; local terminal wait bounded by launch + 7200 s, collection subprocess 900 s; API sockets bounded. Failure stops the pair. Remote work remains governed by its submitted cap if observation fails.
- Collection is COMPLETE but gameplay unhealthy -> inspect actual GPU line, server READY, 25 expected IDs, hour horizon, terminal game states, ending notes, request coverage and fatal server errors. Status alone is insufficient.
- Automatic pictures distract or cost context -> preserve all actual requests and measure input/cutoff counts; no-harm and trace audit decide retention. Not testable with CPU fixtures; accepted research uncertainty.
- Public worker cache variation -> serial arms and frozen settings; comparison remains a screening decision, not a seed-matched efficacy estimate. Startup failure ends the pair under the original stopping rule.

No extra probe, retry, timeout extension, hidden draw or queue mutation is part of execution. Actual GPU identity can only be observed after allocation; requested class is checked before push and the allocated class is mandatory at the control gate.
