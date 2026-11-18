# first-submission-local-20260910-v1
Question: Can the selected frozen reasoning-off NOOA package execute real public LS20 actions through the pinned official Agent loop on this laptop?
Posture: Submission integration qualification.
Contrast: none; controlled loopback responses, not a model-quality test.
Prediction: Three controlled model responses cause three authoritative Solver actions and normal worker cleanup.
Measure: local-v1/requests.jsonl, local-v1/workspaces/*/ipc/transitions.jsonl, stop and worker metadata, local-v1/RESULT.json.
Cap: One local offline LS20 game; three Solver actions; six loopback model requests; 180 seconds outer timeout; zero paid calls and zero GPU quota; monetary spend not applicable.
Stop: Three Solver actions; abort on integration error, request cap, timeout or worker failure; clean up owned worker and loopback server.
Invalid if: Candidate/source hash mismatch, substituted official Agent implementation, lost action/response accounting, failed cleanup, timeout, or external model request.
Result map: Pass -> package the same runtime for bounded Kaggle public-game qualification. Failure -> diagnose the recorded integration boundary and create a new attempt/card after repair. No model or gameplay-quality conclusion.
Outputs: analysis/deployment-probes/20260910-first-submission/local-v1/; original package unchanged.
