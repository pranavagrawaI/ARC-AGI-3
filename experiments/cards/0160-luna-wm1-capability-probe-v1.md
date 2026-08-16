# 0160-luna-wm1-capability-probe-v1
Question: What capability/harness model does Luna form from normal WM1 input, and how does it know?
Posture: diagnostic self-model probe; no gameplay claim
Contrast: none
Prediction: Luna writes capability_probe.md, separating stated affordances from inference and exposing omissions or misconceptions.
Measure: saved note, provider events, worker audit, transcript, summary; classify claims as accurate, omitted, unsupported, or mistaken.
Cap: 1 LS20 action; 14 provider calls; 3600 s; OpenAI route; spend recorded, not preset.
Stop: after one validated action; abort on provider, worker, sandbox, note-write, or host-step failure.
Invalid if: probe is absent from context, note absent/truncated, or normal game input unavailable.
Result map: faithful -> improve utility cue; omissions -> improve visibility; misconceptions -> repair wording; invalid -> repair probe transport.
Outputs: experiments/agents/0159-raw-muse-one-turn/artifacts/wm1-capability-probe-luna-v1/
