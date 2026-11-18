# Kaggle 32k reasoning run

Status: `ERROR` at notebook level. The raw evidence is retained. This is not an engine-startup failure.

The loop arm completed cleanly and passed its local assessment: 20 model calls, 61,766 reported reasoning tokens, 2 valid action batches, 21/21 model accesses reconciled including warmup, and clean lifecycle teardown. It reached no completed level. The arm is useful compatibility evidence, not a gameplay-quality result.

The mindset arm reached 8 completed model calls, 31,325 reasoning tokens, and one valid `UP` action. Its next model request used the 32,768-token total completion allowance. SGLang continued decoding at roughly 32 tokens/second until the server had about 47,452 full tokens (prompt plus completion). The worker's 1,200-second decision timeout fired before a terminal response arrived. Shutdown then found one request still running, so the lifecycle gate correctly marked the arm invalid; the late tokenizer output and FastAPI warning are consequences of that unfinished request.

Therefore the run establishes:

- reasoning is genuinely enabled on the Kaggle RTX PRO 6000 route;
- the repaired loop can complete attributable reasoning/tool/action traffic under this setup;
- a 32k allowance is not compatible with the current 1,200-second worker/HTTP deadline for every long mindset request;
- the mindset arm is not qualified, and no causal comparison is valid.

The next candidate should keep the model/context configuration but lower the per-request completion ceiling or give the worker a deliberately matched deadline, while preserving confirmed retirement before teardown. Do not relaunch this card into the same evidence directory.
