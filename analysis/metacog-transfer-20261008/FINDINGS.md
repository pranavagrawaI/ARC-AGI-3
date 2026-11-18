# Does our metacognition research transfer to the incumbent (dfranzen M2, Duck lineage)? — 2026-10-08

Data: dfranzen-m2-control v1 (public-25, 107 levels), `../m2-winners-20261001/runs/out-control`. Scripts here; all offline, zero GPU.

## Observations
1. `stuck_tokens.py`: 1.4M of 4.3M generated tokens (33%) go to the final level a game never solves. The scheduler
   abandons a game (`gave_up`) after ~100-140K tokens on one level; a solved level averages ~27K (2.9M / 107).
   Six games die at L0/L1 (sk48, g50t, lf52, tn36, s5i5, bp35). Wall share (`stuck_time.py`) is 37% but wall is
   scheduler-confounded (stuck games get starved), so tokens are the measure.
2. `stuck_modes.py` (marker rates per 100K thinking chars, crude regexes): stuck levels are not one failure.
   - Goal doubt is already HIGHER when stuck: lf52 33 vs 12 on its solved level, tn36 20 vs 3, ka59 26 vs 20.
     The model questions itself without being asked. Consistent with our nulls (self-question 74 vs 72 over 75
     episodes; external questioner entrenches).
   - s5i5 L2: the model writes a simulator, fits it to every observed transition ("sim6 matches ALL real transitions,
     0 mismatches"), runs BFS/best-first, and dies on infrastructure: NameError on its own constants (module-level
     data is not retained), `time` import blocked, search state limits under the 30 s call cap, budget exhausted.
   - lf52 L2 (`churn` probe, concept-level): the same three goal hypotheses are reopened across ~85 actions
     ("one peg per panel" at 24 decision points, "clear all pegs" at 29, "box is the hole/goal cell" at 14).
     Generation is abundant; closure is absent. The lexical proxy in churn.py says 4% restatement — it misses
     paraphrase; trust the concept count, but it is one game.
   - Checked and rejected: lf52 L1 was genuinely "reduce to 1 peg", so this is NOT a repeat of the Qwen3.8
     lossy-level-up lesson.

## What it supports
The binding metacognitive deficit on stuck levels is selection and retention, not generation: hypotheses are not
closed by decisive tests and closed ones are not kept; fitted simulators do not persist. Our questioner/doubt line
targets generation, which is already saturated — do not port it. Our Tycho/NOOA line targeted retention (executable
world model, prediction check) but as extra agents, which cost 2-12x tokens (pair-1). The transferable form is
tools inside the actor's own sandbox: zero extra calls.

## Uncertain
n = 1 control pass; mode classification by regex plus reading 2 games. Size of gain unknown; identical-run sd is
~11 levels per 25-game pass, so levels alone cannot decide a single-pass arm — read mechanism counters.
