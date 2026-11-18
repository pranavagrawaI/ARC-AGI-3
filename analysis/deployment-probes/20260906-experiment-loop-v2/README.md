# Isolated experiment-loop candidate

Start with RESULT.md and COMPARISON.json. The queued Kaggle package remains unchanged.

experiment_agent.py adds the regional experiment protocol to the frozen q6 MdArcSolverAgent. state_delivery.py replaces stale queued snapshots with the host's current state. headless.py wires those changes together. actor-1.py selects the experimental variant; actor-0.py selects the original actor with only the current-state dispatcher correction. The live runner uses the existing Kaggle-style worker and a bounded local OpenRouter relay.

Tests and offline scoring are safe to rerun using RESULT.md's commands. probe.py is an as-run artifact and refuses reuse once requests.jsonl exists. A further live trial must use a new output directory and precommitted card. Do not edit SOURCE-PIN-covered files to reuse an old trial identity.

The public experiment API:

1. experiment(hypothesis, alternative, action, region, expect_change, rationale) records a specific prediction and commits one legal action.
2. experiment_result() returns measured changes from the actual successor transition.
3. close_experiment(conclusion, next_hypothesis, next_test) requires a correct binary conclusion and revision when contradicted before another experiment.

A supported regional prediction establishes only that the predicted region did or did not change. It does not establish object identity, motion direction, reversal or cause. Those claims require their own discriminating measurements.
