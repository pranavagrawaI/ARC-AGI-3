# 0159-muse-provider-qualification-v17
Question: Which Muse endpoints execute the image plus two-tool NOOA transport within bounded latency?
Posture: qualification, not gameplay or model-quality evaluation
Contrast: DeepInfra BF16, Together, Fireworks, Phala; same model, prompt, image, tools, limits
Prediction: at least two endpoints complete both required tool calls within 120 s each.
Measure: per-attempt JSONL: provider, latency, finish reason, tool calls, usage, error, verdict.
Cap: 4 providers; 2 calls each; 120 s/call; 10 min wall; 256 output tokens/call; unknown spend.
Stop: test every provider once; abort on credential failure or cap breach.
Invalid if: selection is unenforced, schemas differ, raw metadata is absent, or timeout does not cancel.
Result map: >=2 pass -> enable ordered fallback from passers; 1 passes -> retain it and diagnose resilience; 0 pass -> stop before ARC.
Outputs: experiments/agents/0159-raw-muse-one-turn/artifacts/provider-qualification-v17/
