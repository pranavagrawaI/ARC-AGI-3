# Local preflight

Final manifest: 5074227fe57be65d92e2a4765347a27514513ea833a3d5e730b1c9d9d52e603e.

Actual adapter mapping (tycho_nooa/protocol.py): ACTION1=UP, ACTION2=DOWN, ACTION3=LEFT, ACTION4=RIGHT. Synthetic successor rules are stored only in the host runner. The Actor sees the current grid and receives its successor after submission.

All four final local runs use the actual adapter, isolated worker, production solver methods, and local HTTP endpoint. Each preflight directory retains received requests, mock responses where applicable, worker logs, STARTED identity, and RESULTS.json.

| Case | Receipts | Mock admissions/responses | Worker records | Result |
| --- | ---: | ---: | ---: | --- |
| success | 2 | 2/2 | 2 | UP then DOWN; one call per decision; second cell asserts measured no-op |
| call_limit | 3 | 2/2 | 2 | Third call refused; worker retired |
| deadline | 1 | 0/0 | 0 | Expired admission clock refused request; worker retired |
| rejection | 1 | 0/0 | 0 | Explicit refusal; worker retired |

Both success prompts passed exact demonstration inclusion and diagnostic marker checks and omitted the ordinary arc_skill and mandatory world-model system instructions. Worker startup asserts no summarizer is installed. Mock usage and worker records both show reasoning zero. No helpers or procedures were created. The cleanup-generated final transition must not be interpreted as a second simulated environment step: only the first successor was applied by this two-decision screen.

Focused fixture, real-client, and lifecycle suite: 24 passed in 15.95 seconds. Earlier preparation failures are retained in separately named preflight directories. No provider contact occurred during preflight.
