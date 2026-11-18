# q6 summary boundary repair

Question: Can the q6 Actor summarize note-containing history and continue gameplay while preserving note integrity and exact Actor/summarizer request accounting?

Basis: The parent q6-prediction-feedback-20260906 run failed after 19 actions at its first attempted summarization. History serialization escaped the expected note event; the shared Actor-format guard aborted before dispatch. Its frozen identity and raw evidence are preserved. This successor changes the integrity boundary and related coverage selection only; prediction workflow and baseline FP8 inference are inherited.

Local qualification: Reproduce the old failure with the real NOOA runtime and a fake model; replay the captured Kaggle request; exercise summarize, collapse, and resumed Actor calls; verify quoted notes, retries, repeated compression, and archived IDs; reject deliberately changed input/readbacks and intact decoys. Keep summarizer recording and reasoning rejection. Rebuild/extract the notebook and test its source. These checks do not measure summary semantic quality or GPU performance.

Prospective live cap: At most one private push to pranavagrawai/q6-summary-boundary-20260906, one ls20 game, 100 individual environment actions, 7200s gameplay admission from the first playable action. Pending decisions may finish within the inherited 900s worker deadline. Launcher envelope 8400s, Kaggle session 10800s, maximum 3 GPU-hours. Baseline FP8, thinking disabled, 4096 output tokens, 180s model request timeout, 1200s startup timeout, ARC_STALL_REVIEW=0. Zero automatic retries or competition submissions. Recheck quota before any push. Local qualification alone does not initiate a launch.

Stop: WIN, action cap, or gameplay admission horizon triggers normal cleanup. Worker failure, hard timeout, integrity/reasoning violation, or missing/extra requests fails qualification. Do not cancel a model call to manufacture clean completion.

Evidence: Source/launch hashes; transitions, actions, notes, model predictions and feedback; integrity.jsonl including SUMMARY_INPUT_VERIFIED; model_calls.jsonl with explicit summarizer roles; launcher status, server requests, and game/session stop records. Zero completed summarizer calls does not establish zero attempts: inspect failures and request guards.

Result map: Clean compression and resumed play with reconciled calls qualifies the live boundary only. Repeated contradictions or weak summaries require separate solver-quality investigation. No live compression leaves that path unqualified. Any infrastructure failure requires diagnosis before interpreting solver progress. Encoder matches remain consistency evidence rather than independent game truth.

Status: Local repair and tests requested; no successor launch performed in this work. LAUNCH.json binds the prospective notebook, metadata, card, and tools.
