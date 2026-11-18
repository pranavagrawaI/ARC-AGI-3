# m2-evidence-smoke-1
Question: Does the m2_evidence treatment (dfranzen + m2_basics fixes + m2_memory data API + append-only evidence record) run end-to-end on the real Kaggle serving stack, and does the real model write evidence entries when it is free to?
Posture: de-risking before the pair; operational feasibility only.
Contrast: none. No score or level claim; feasibility and delivery only.
Prediction: Session COMPLETE. The log shows "M2 evidence installed" and "priority gate active". All 3 games (lf52, s5i5, ft09) reach a terminal state (won, or gave_up at the 870 s cap) with no harness exception. At least one game writes artifacts/*_evidence.json with ≥1 entry. The transcripts show tool calls parsed by the real model's tool format.
Measure: output/dfranzen-m2-control.log (install line, solver config, tracebacks); output/benchmark.json game_runs[].state and solver_note; output/artifacts/*_evidence.json (entries, reshows); output/transcripts/*.txt (record_evidence calls, "Your evidence record" blocks, python tool errors mentioning evidence).
Cap: 1 session, 1 RTX PRO 6000, hard timeout 2,400 s (0.67 GPU-h) charged against the 6.6 GPU-h experiment ceiling; 3 games, 870 s each (priority-scheduler share for 3 games), 1 pass; no action cap; no money; no submission.
Stop: Session ends at its caps. No retry under this card. A failed push or acceptance-unresolved push is charged at its hard cap and not retried.
Invalid if: the install cell did not run or hash-rejected the bundle; the session ran on a different machine shape; outputs could not be downloaded.
Result map:
- Operational pass AND ≥1 evidence entry → launch pair-1 under its frozen card.
- Operational pass AND 0 entries in all 3 games → do not launch pair-1. The treatment was not taken up; read the transcripts for why (prompt placement, cost, tool friction) and bring a redesign decision to Pranav. pair-1 would test only m2_memory.
- Harness failure attributable to the overlay (traceback in our code, sandbox protocol error, crash on record_evidence/evidence/digest) → no pair. Fix offline, re-gate, new smoke card with a new run ID.
- Infrastructure failure outside the overlay (queue, Kaggle ERROR before install, serving start) → diagnose from the log. A rerun needs a new run ID and card and is charged to the same 6.6 h.
Outputs: solvers/m2_evidence/experiment/smoke-1/{launch-attempt.json, launch-response.json, status-history.jsonl, output/}; budget.json.
