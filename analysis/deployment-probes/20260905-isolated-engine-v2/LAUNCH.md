# Competition-bound retry launched

Kaggle accepted version 1 of `pranavagrawai/qwen38-isolated-engine-20260905-v2`
on 2026-09-05, approximately 13:50 UTC. Status after push: RUNNING.

Both the installed CLI request-construction test (transport mocked) and pulled
provider metadata confirm `competition_sources=[arc-prize-2026-arc-agi-3]` and
`machine_shape=NvidiaRtxPro6000`. The CLI request test confirms 3600-second cap,
private/offline configuration. All pulled notebook code cells match v1; local
notebook bytes are also identical. Only metadata and candidate identity change.

The user identified the competition binding as the RTX PRO 6000 access
requirement. V1 had incorrectly removed it while isolating the engine benchmark.
The competition attachment does not execute games or submit a competition entry.

Actual device identity and three-arm results remain pending runtime evidence.
The existing hardware gate remains intact. No further automatic retry.
