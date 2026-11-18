# Synthesis review and authorship

Five native subagents independently produced the early-history, runtime, inference, discovery, and controller appendices. The coordinator produced research-decisions, the source inventory/board snapshot, the main synthesis, and its verification scripts. All writes from this task are within `analysis/what-we-learned-20260909`; original solver and experimental evidence files were not edited by this task.

| Slice | Agent | Session ID |
| --- | --- | --- |
| Early history | Jason | 01a086b6-172c-7253-a375-de04e74cdeaa |
| Runtime | Boole | 01a086b6-1759-7182-838f-60a5fb57abf5 |
| Inference | Carver | 01a086b6-179f-7f92-a133-620939bfed87 |
| Discovery | Maxwell | 01a086b6-17e5-7c90-9420-eacc41e41773 |
| Controller and final synthesis review | Wegener | 01a086b6-182c-73d3-972d-b22434807acb |

The coordinator read all appendices, reconciled source/report discrepancies, added the explicit successes and later procedural/serving outcomes, qualified symmetry's trace-based causal language, and recorded concurrent source-report updates without altering the initial inventory.

The controller investigator then independently reviewed the integrated report for material incorrect claims or lost qualifications, particularly supplied-plan execution, recorded non-use, level completion, causal attribution, and the concurrent audit update. Result: no material issue found within that scope. This was a synthesis review, not an independent replay of every historic experiment.

The coordinator's raw checks cover q6 summary-boundary and both controller arms, plus the old supplied-route ceiling. Subagents separately checked additional raw ledgers identified in their appendices. Historical test counts are reported evidence; this task launched no model, GPU, or game experiments and reran no historical qualification suites.

Reproduce collection with `python3 collect_sources.py NEW_DIRECTORY`, which refuses an existing destination. Reproduce link/identity/selected-outcome checks with `python3 verify_report.py NEW_JSON_FILE`, which refuses to overwrite a result. Existing output files are successive immutable checks, not one rewritten verdict. The final result is linked from REPORT.md.

Limits: discovered documents are not all substantively reviewed; some old original raw directories and external recovery archives are missing. The board snapshot ends at cursor 385. Concurrent later report revisions accepted by the coordinator are separately archived with text and hashes in sources/reviewed-source-updates.json.
