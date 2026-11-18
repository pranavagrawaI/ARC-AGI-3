# Aborted baseline diagnostic

Four real OpenRouter CodeAct requests cost $0.011016352. No action submission or candidate start occurred. The baseline first failed to use async game_states.get correctly, then generated an unbounded loop that appended coroutine objects without awaiting them. Worker RSS reached 21,324,088 KiB at inspection, about 21 GiB, and the supervising agent terminated that worker with SIGTERM after 95.7 seconds. The relay stopped without launching another arm.

This exposes a solver state-access defect and insufficient local resource bounds. It supplies no comparative evidence about prediction feedback. The pre-existing 60-second cell timeout is advisory for the in-process backend and did not prevent allocation. Recorded requests/responses retain the exact failing code. PINNED-SOURCES.tar.gz preserves the exact source hashes checked before this run; INCIDENT.json records the intervention.

SCORE.json evaluates saved helpers even for unstarted fixtures. `accepted_submission: false` applies to all four; their scores are not completed Actor results. The failed baseline helper remains unchanged and predicts every held-out band edge incorrectly.

Next identity applies the same synchronous, actionable queue-access refusal to both arms, retains exact-history access, and adds 1 GiB RSS supervision, 4 GiB address-space and 90-second CPU hard limits. That is a changed experiment, not a retry within this card.
