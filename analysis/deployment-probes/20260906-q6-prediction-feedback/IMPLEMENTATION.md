# q6 solver feedback candidate

The candidate is isolated in deployment/kaggle/q6-prediction-feedback-20260906. The hardware-qualified quick-host package and other agents' inference work are unchanged.

Implemented:
- Freeze the Actor-authored encode/predict source before an admitted action, then measure field discrepancies under that same encoder on actual successor frames.
- Require a successful prospective check of the same helper version before batching; single-action exploration remains available when a model is absent or unresolved.
- Put a bounded partial-repair/test workflow in the Actor context. It is an instruction, not a guarantee or forced correctness verdict.
- Replace the exposed asynchronous dispatcher queue getter with an immediate error directing the Actor to synchronous latest_state()/trajectory(). The old getter caused a live 21GB coroutine accumulation.
- Bound CPU worker virtual memory at4GiB. Bound automatic Python helper evaluation with a1s alarm, restoring the prior handler. Native code may defer signals; outer worker limits remain necessary.
- Tolerate incomplete final JSONL appends; complete malformed rows remain errors.
- Stop gameplay admission gracefully at7200seconds or100individual actions; retain a900second worker envelope,8400second launcher envelope and10800second Kaggle cap.
- Replay the final pending prediction prefix after downloading, separately marked not delivered to the Actor. No scoring code mutates the raw evidence.

Local checks include real CodeAct mismatch -> batch rejection -> helper repair -> successful single-action prediction -> batching reopened, raw captured contradiction replay, partial-tail/corruption distinction, helper loops/exceptions, source freeze, Actor/summarizer request accounting, history/state dispatch, and actual worker clean deadline shutdown. Monitor tests cover terminal outcomes and single-download behavior.

Live diagnostic outcomes and adoption disposition belong in RESULT.md. The longer run has no matched gameplay control: report progress and model consistency separately, without treating either as proof of causal improvement.
