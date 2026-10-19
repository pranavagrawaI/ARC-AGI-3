# F1x production-path probe, 2026-10-18

Question: Does the exact scheduled F1x solver and server operate with production
flags, minimal diagnostics, no request snapshots and 110 unique games over the
competition HTTP client?

Posture: Diagnosis and operational qualification. Prepared locally; no GPU launch
or queue change is authorized by this card. Ask Pranav before spending GPU quota.

Candidate: `pranavagrawai/f1x-production-probe-20261018`, intended version 1.
Parent: exact F1x screen-2 v1 notebook SHA256
`826dfdc2a483a786bdfe888ad9b9422630bc4c615a187b34bdc9762975fb7df9`.
The generated notebook hash is in PREPARATION.json and must be frozen before launch.

Contrast: None. This is a bounded feasibility diagnostic, with no gameplay efficacy
or attribution claim.

Prediction: All 110 cloned public games initialize through the competition client;
the unchanged F1x solver and SGLang server generate tokens and execute native actions;
the notebook exits with preserved progress and no crashed games.

Measure: Source and metadata readback, live/terminal kernel logs, serve.log,
probe-config.json, probe-progress.jsonl, probe-exception.txt if present,
probe-final.json, probe-gates.json and the PROBE_OPERATIONAL_PASS marker.
Require normal remote completion, 110 unique initialized games, no crashed games
or analyzer-failure notes, actions in at least 10 games and positive reported
generated tokens. These gates measure operation, not score. Cancelled games at
the explicit deadline are expected.

Cap: One launch; one RTX PRO 6000; 2,700 GPU-seconds including startup;
900 seconds of play after local gateway startup. Zero training, zero paid compute,
zero hidden submissions and zero automatic retries. Check current GPU quota and
actual GPU sessions before admission.

Stop: Normal completion, platform hard cap, setup failure, benchmark exception or
failed operational gate. Preserve every failed attempt. Do not relaunch on failure.

Invalid if: Source drift; different serving/sandbox/model settings; wrong production
flags; missing simulator or wrong game count; missing final evidence; no model
generation/actions; crashed games; timeout before normal completion.

Result map: Failure with evidence -> isolate the failing boundary before another
F1x hidden draw. Valid pass -> shared production paths survive this bounded probe;
Kaggle-specific gateway, hidden content and long-duration failures remain possible.
Missing evidence -> inconclusive; no automatic retry or promotion. Any later hidden
draw needs its own existing or renewed authorization and exact artifact binding.

Outputs: This directory and its kernel/ subdirectory; remote artifacts, if a launch
is subsequently approved, go to output/. `probe.patch` contains all executable edits.

Limits: Public games are cloned to 110 unique IDs. The local server exercises the
HTTP client and shared scorecard lifecycle, not Kaggle's gateway deployment or hidden
games. F1x omits the F1-only prompt/counter changes, so success cannot explain D9 by
itself. This short probe cannot establish nine-hour reliability.
