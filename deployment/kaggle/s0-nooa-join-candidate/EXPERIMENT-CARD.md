# s0-nooa-join-local

Question: Can the official MyAgent / protocol-v3 seam drive the q3c NOOA chassis (file-IPC `states.jsonl` → `submit_actions`) so one public observation becomes one typed `ActionDecision`?

Posture: Execution. Feasibility of the join, not gameplay competence.

Contrast: none

Prediction: A stub NOOA agent that writes one `submit_actions(["UP"])` on the first state produces `ActionDecision(action="UP")` through `LocalSolver.decide`, and a two-action batch is drained one official turn at a time without a second state line.

Measure: `deployment/kaggle/tests/test_nooa_join.py` — decision.action, states.jsonl line count, finish/close.

Cap: local CPU only; no Kaggle push; no live model; no ARC environment action. Wall 120 s for the pytest file.

Stop: success if the stub join tests pass. Futility if the worker cannot return a typed decision from IPC. Abort if a test would require OpenRouter, SGLang, or a Kaggle kernel.

Invalid if: the stub agent is not the process behind `actions.jsonl` (worker invents the action), or MyAgent is bypassed for the adapter test.

Result map:
- positive → next is a fake-LLM MdArcSolverAgent turn in this candidate, then a notebook that embeds the same worker. No Kaggle until that notebook's first Phase A is "scorecard produced + one committed action".
- negative → repair the worker; do not open q5 or a new chassis.
- ambiguous → preserve traces under the session_root the test prints; do not launch Kaggle.

Outputs: pytest stdout; worker session_root `ipc/states.jsonl`, `ipc/actions.jsonl`, `worker-stderr.log`.
