# Reasoning-enabled designer-player candidate

Separate successor to designer-player-20260907; old evidence unchanged.
Current interface repair: `last_transition()` exposes measured changes, with
exact before/after grids available on request. `preflight_model(actions)` runs
the same JSON-normalized preparation as submission without committing anything.
The Actor instructions distinguish measurements from hypotheses and no longer
promise absent diff/animation fields. Old trajectory keys remain compatible.
Both notebook arms enable thinking/reasoning and allow32768 total generated
tokens per request. The real client enforces that allowance and rejects disabled
or conflicting reasoning flags. Reported reasoning counts are not fabricated
from character counts; presence is tracked separately when usage is absent.
Natural stopping remains enabled; the experiment measures how much of the larger
allowance the model actually uses. The agent is told the mounted model's verified
262144-token context capacity, so the 32768-token output ceiling does not consume
the entire advertised context.

The recorder retains reasoning-bearing tool calls and still verifies exact
history/summarizer input. The prior empty-content retry is disabled because a
null-prose response with reasoning plus a tool call is valid CodeAct output.

Use prepare_candidate.py, then prepare_candidate.py --check. Run verify_candidate.py
with the installed Kaggle CLI Python to inspect the upload request without network.
Read QUALIFICATION.md before launch. No upload or launch is part of preparation.

The notebook compares loop against loop+designer-player framing, not thinking-on
against thinking-off. Thirty actions and1800s admission per arm;1200s pending worker;
1200s Actor HTTP;10800s session. FP8 checkpoint and kernel settings unchanged; MTP parked.
OpenRouter live tests establish compatibility only, not Kaggle numerics or speed.

Lifecycle repair: unhandled generation errors are terminal, not silently retried.
An interrupted LocalSolver decision terminates its worker before releasing model
admission; unconfirmed retirement blocks subsequent admission. The local paid
probe closes an independent clock-checked HTTP gate before cleanup and bounds
relay shutdown. These fixes do not shorten reasoning or add retries. The old
failed run remains unchanged; a new live test is not authorized by local repair.
