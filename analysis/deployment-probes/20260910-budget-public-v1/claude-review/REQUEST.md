# Independent Claude verification request

Task ID: budget-public-claims-review.
User explicitly requested a Claude agent through Superset to verify all of the
coordinator's latest claims and proposed next tests. Be independent: neither the
coordinator's PASS nor its projections are evidence. Read applicable workspace
instructions. Consult raw records and the exact packaged/live source.

Scope: read-only review of source and preserved evidence. You may create your
review and reproducible read-only analysis scripts ONLY inside
analysis/deployment-probes/20260910-budget-public-v1/claude-review/.
Do not edit source, notebooks, original cards, logs, or existing reports. No
Kaggle launches, uploads, competition submissions, model calls or paid experiments.
No nested agents. No git commits/pushes. Preserve unrelated work.

Read first:
- analysis/deployment-probes/20260910-budget-public-v1/EXPERIMENT-CARD.md
- analysis/deployment-probes/20260910-budget-public-v1/RESULT.md and audit.py
- Raw qualification/, source/, remote-source/ under that same run directory.
- deployment/kaggle/submission-20260910-v5/{build.py,play.py,README.md,BUILD.json}
- deployment/kaggle/submission-runtime-20260910-v3/, especially headless.py,
  shared_local_solver.py, nooa_worker.py, chassis/solver_agent.py and arc_llm.py.
- analysis/deployment-probes/20260910-budget-handling/REPORT.md for local evidence.

Verify these claims, independently recalculating from raw evidence:
1. Remote notebook version 4 completed a public-only run of LS20/S5I5/CD82;
   no competition submission by coordinator. Frozen operational PASS is valid
   under the literal card, no recovered-OOM or request-failure exceptions.
2. Session times LS20 92.627740502 s, S5I5 83.839930341 s, CD82 78.54498957 s;
   actions 1/3/1, completed Actor calls 11/12/11, zero levels. All acknowledged
   model-admission stops; 34 completed calls + 1 warmup = 35 server requests;
   clean shutdown, no OOM/deleted-state markers; no summaries/reasoning-on calls.
3. Model-call wall times 88.404761352 / 79.29067919 / 74.563944008 s (LS/S5/CD),
   total 242.25938455 of 255.019576933 gameplay seconds (~95%). 34 calls:
   mean 7.125276 s, median 5.643897 s, p90 13.729653 s, max 28.554695 s;
   8539 output tokens, prompt range 13165–29906. Specify percentile convention.
4. Notebook last-log time 626.327837106 s; outer runtime 584.148 s; gameplay
   255.019576933 s. Difference notebook minus gameplay ~371.30826 s is combined
   non-game overhead, NOT all model loading. Are clock boundaries and attribution
   appropriate? Should any number be called an approximation rather than total?
5. Current competition launcher: 25200 s, actual play budget 25140 s (6h59m),
   outer subprocess 27600 s. It does not exploit all of the user's assumed
   9-hour allowance. Verify the actual invoked runtime, not just unused defaults.
6. Assuming 110 games and unchanged short-run action throughput:
   25140 gameplay seconds -> 228.545 s/game, ~492.903 actions total (~4.48/game).
   32400 minus observed 371.30826 outside-game seconds -> 291.170 s/game,
   ~627.965 actions total (~5.71/game). Explicitly labeled arithmetic illustrations,
   NOT forecasts; excludes additional safety reserves. Assess whether useful or
   materially misleading, including quota vs session runtime, per-game reserves,
   history growth and startup amortization. 23 of 34 calls preceded first actions
   (6+8+9, including calls producing those actions). No score forecast licensed.
7. Solver description: NOOA MdArcSolverAgent using Qwen3.8-27B-FP8 via SGLang;
   sequential games; grid/text default not image vision; CodeAct Python/NumPy
   analysis, notes/helpers/hypothesis registration; only submit_actions advances
   gameplay; observations/predictions checked; batches gated on predictive
   agreement; contingent controller can execute one matching continuation without
   a model call; routine reasoning off and background summarization disabled;
   each model call guarded by the new admission deadline. Distinguish available
   capabilities from actual use in this run, and qualify any inaccurate wording.
8. Proposed tests (NOT launched): endurance, competition-sized per-game budget,
   local injected slow HTTP/error/stalled-tool failure containment. Proposed next
   public run: 10 min model admission/game on the same 3 games, action cap raised
   10->200, same model/solver sequential, request/cleanup reserves, 1h global cap.
   Compare first/second 5min windows for actions, calls, context, latency/memory.
   Assess adequacy and feasibility; identify stronger cheaper discriminating tests
   if warranted. Does the proposal change only the intended limits? What exact
   cap/configuration changes would actually be required? Do not implement them.

Deliver REVIEW.md with claim-by-claim status (verified / corrected / unsupported /
conditional), file:line evidence and reproduction details; severity-ranked
findings; corrected calculations; recommendation for the next test. Distinguish
what you ran from source inspection. Do not silently change the precommitted card.
If provenance cannot establish something (e.g. historic remote version identity),
say so rather than upgrading a coordinator assertion into independent evidence.

End your terminal response with:
SUPERSET_WORKER_DONE
task: budget-public-claims-review
summary: <one-line outcome>
files: <review paths>
checks: <commands and outcomes>
handoff: <corrections or none>

If blocked instead end with:
SUPERSET_WORKER_BLOCKED
task: budget-public-claims-review
reason: <specific blocker>
needs: <required input/access>
