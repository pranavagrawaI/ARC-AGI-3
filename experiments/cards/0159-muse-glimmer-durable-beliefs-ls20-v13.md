# 0159 Muse durable beliefs LS20 v13
Question: Do model-authored durable notes help Muse retain useful knowledge in a 44-action LS20 run?
Posture: capability probe; evidence preservation.
Contrast: v12 codeact plus only run-local NOOA-style notes and a semantic-curation instruction.
Prediction: Muse writes then reads action-semantic or mechanics notes; levels completed does not decrease from v12.
Measure: `summary.json` levels/actions/stop; `provider-events.jsonl` knowledge events; exact `knowledge/*.md`.
Cap: 44 ARC actions; 14 provider calls/turn; one online game.
Stop: level complete, cap, failure, or operator abort.
Invalid if: wrong model/provider/source, scripted client, unenforced sandbox, uncertain ARC action, or host-authored semantics.
Result map: useful later-read notes -> retain and next add executable model; unused -> inspect interface; ambiguous -> fix technical interface and use a new card.
Outputs: run artifact directory.
