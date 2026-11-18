# Kaggle designer-player interface-v2 / 32k reasoning qualification

Question: On the actual ARC-AGI-3 Kaggle RTX PRO 6000 route, does the current interface-v2 solver produce attributable reasoning, valid actions, and useful progress, and does designer-player framing change its behavior relative to the repaired loop alone?

Posture: bounded integrated qualification and exploratory comparison. This is not deployment promotion or a competition submission.

Contrast: fresh public `ls20` loop arm followed by mindset arm. Both use the identical immutable Qwen3.8-27B FP8 checkpoint, baseline serving path with MTP off, sampling defaults, reasoning flags, 262144-token advertised context, 32768 total generated tokens per request, natural stopping, and game/time limits. Only designer-player context differs.

Prediction: both arms establish real reasoning and commit at least one valid action. The mindset arm should more often state objective hypotheses, designer-intent constraints, unresolved questions, and discriminating tests instead of repeatedly modelling local motion. The larger allowance should reveal the model's natural reasoning demand without forcing it to fill 32768 tokens.

Measure: effective request configuration; prompt, completion, reasoning and tool-call token evidence per model call; stop reasons; model wall time; request/server reconciliation; decision experiments; hypotheses and unresolved questions; valid action batches and transitions; interrupted queued actions; grid changes; level progress; lifecycle and containment. Inspect each CodeAct cycle rather than treating aggregate reasoning tokens as quality.

Cap: one 10800-second notebook; two arms; 1200-second server startup each; 3000-second launcher timeout each; 1800-second gameplay admission each; 1200-second pending-worker and Actor HTTP limits; 30 actions per arm; compatibility warmup HTTP remains 180 seconds. No automatic retries, model replacement, MTP, submission, automatic promotion, or relaunch.

Stop: normal win/action/time boundary. Any manifest, source, configuration, containment, lifecycle, or request-accounting failure stops before the next arm. A technical failure is retained and diagnosed without automatic relaunch.

Invalid if: reasoning is disabled or unestablished; mounted model, competition, GPU, candidate contract, request configuration, or physical request accounting is unverified; an admitted request lacks attributable terminal disposition; lifecycle or containment fails; arm inputs differ beyond the declared prompt treatment; or evidence is overwritten by a retry.

Result map: valid reasoning plus valid actions establishes Kaggle compatibility for this candidate. Better equal-budget objective discovery or progress by mindset supports a larger matched evaluation; no difference is inconclusive, not proof the framing is useless. Hitting 32768 indicates budget pressure, not reasoning quality. Shorter natural use indicates headroom. No valid action or purely repetitive reasoning identifies a solver/control-loop target. Technical failure triggers diagnosis only. No result promotes to submission automatically.

Outputs: `/kaggle/working/designer-player-loop/nooa-join-public3` and `/kaggle/working/designer-player-mindset/nooa-join-public3`, including raw server/lifecycle/IPC, request counts, and `ASSESSMENT.json`; local `LAUNCH.json` binds notebook, card, source contract, competition, and GPU identity.
