# Oracle-hint readout, corrected, and the A1 failure reading (2026-10-11)

## Correction to the readout
Eight attempts were flagged as solve/terminal disagreements: lp85 L7 ×5, tu93 L8, cn04 L5, sc25 L5. Every one is a
game's **final** level. Completing it ends the game as WIN, and no `solved` event is emitted. `analyze.py` now counts
solve = solved event OR benchmark `won`; the pre-fix readout is kept as `evidence/readout-pre-won-fix.json`.

| | p0 (A0) | p1 (A1 goal) | p2 (A2 goal + route) | d1 [90%] | d2 [90%] |
|---|---|---|---|---|---|
| before | 0.29 | 0.46 | 0.85 | +0.17 [0, 0.35] | +0.56 [0.42, 0.69] |
| corrected | 0.33 | 0.50 | **1.00 (26/26)** | +0.17 [0, 0.35] | +0.67 [0.56, 0.79] |

The verdict stays INCONCLUSIVE: d1 lands between 0.15 and 0.30. Perception/interface is ruled out, because the
model completes every level once it is handed the route.

## Time, not just rate
Solve times among solves (1,200 s cap):

| arm | median | quartiles |
|---|---|---|
| A0 | **1,003 s** | 725–1,082 s |
| A1 | **607 s** | 361–967 s |
| A2 | **144 s** | 101–396 s |

A0 is right at the cap edge, so p0 is largely censored by the time cap. Two levels where A0 solved 2/2 and A1 0/2
(tu93 L3, tn36 L1) had A0 solve times of 626–1,054 s. The hint arm's effect is better read as a speed-up (hazard) than
as a rate change. In-run, by contrast, a game has 7,254 s.

The card's "second hint replicate" is the wrong next spend: the A1 replicates agree on 24 of 26 levels, so the d1
interval is set by level-to-level spread, not replicate noise.

## Why the 26 A1 attempts failed
All 26 ended `gave_up` at the 1,200 s cap after 72–106K generated tokens (≈ 80 tok/s per stream). Tool errors were
few (0–7 per attempt) and tool timeouts nearly absent. Each attempt spent 170–300K characters of reasoning, with
single thoughts up to 40K characters (~2 minutes of decode).

Classification of the model's final thought at the cap against the hint and the verified route (one reader, not
blind; `digest.py <attempt>` reproduces each):

**Mechanics unresolved, 17.** The goal was known, but how actions or other agents change the board was still being
worked out:
cn04-L5-r1 (how selection reveals pieces), ft09-L4-r2 (concluded the hint's rule is unsatisfiable on its reading of
the clues), ka59-L3 ×2 (how to select and move the plus), lf52-L2 ×2 (shuttle and corridor alignment), sc25-L4-r2 (how
the pip grows), sc25-L5-r2 (how to clear the blocker), su15-L3-r1 (what the bar means), tn36-L1 ×2 (what the switch
program does), tu93-L3 ×2 (the red guard lunges when you enter its column; the hint said only "kills if you meet
it"), tu93-L8 ×2 (enemy movement rule), vc33-L4-r1 (what "at the gap" means), wa30-L2-r1 (how many slots/crates).

**Executing a plan when time ran out, 9:**
cn04-L5-r2 (3 of 5 pieces joined), sc25-L4-r1 (firing the clearing beam), sc25-L5-r1 (walking the planned path),
sk48-L0 ×2 (simulator-verified 5- and 2-move finishes), su15-L3-r2 (writing a planner), vc33-L4-r2 (pressing the
candidate target), wa30-L1-r1 (slots filled, waiting on completion), wa30-L1-r2 (17-move plan, 44 s left).

Once the rules are right, planning is not the bottleneck. The model writes BFS and simulators over its own model
of the game unprompted (tu93 L3: correct orange-patroller timing, correct shortest routes). It fails where that
model of the game's rules is wrong or unfinished. A goal hint does not supply the rules; a route makes them
unnecessary.

Tested and null: action-rationing language ("costs 1 action", "budget", "fewer actions", ...) occurs at the same rate
in failures and solves in every arm (1.6–2.0 per 10K reasoning characters).

## What this says about levers
- The binding deficit on these levels is **inducing the game's rules** (action effects, agent dynamics), paid for in
  decode time. Goal judgment is second (+0.17, an upper bound that includes restored cross-level context; A1's 0.50
  roughly recovers the 0.56 in-run rate).
- The class of "planning/closure tools" the card mapped to EXECUTION should be narrowed: route search already exists in
  the model's own code; what is missing is a correct transition model, sooner.
- Of the 26 failures, the 9 "executing" ones are mostly artifacts of the 1,200 s cap; in-run they would likely land.

## In-run reading: the three full dfranzen control runs (`inrun.py`)
53 stuck (run, level) segments on the 40 hard levels (`inrun.py table`, `evidence/inrun-table.json`). 21 sat on one
level for ≥ 45 minutes; I read all 21 (`inrun.py read <game> <level>`: thinking blocks sampled across the segment
plus the last three). The shorter ones mostly entered the level late and ran out of game time.

**Same failure as the sprints.** In all 21, the model is still inducing the rules after 50–110 minutes, and often
finds a decisive mechanic only in its last few blocks: ft09 L4 (clicking a clue cell recolours the board), sp80 L1
("51 strict wins found" in the last block), s5i5 L1 (a 31-action plan in its final blocks), tr87 L1 (glyph extraction
finally correct), tn36 L1 (cards swap). Secondary patterns:
- wrong carryover from the previous level (tn36 L1 keeps matching the left panel, the rule its own account of
  level 1 gave, which is false);
- a constant tax of self-inflicted parse bugs: coordinate offsets, detection failures, unretained helpers. A crude
  regex flags ~15% of thinking blocks in both stuck and solved segments, so it is uniform, not the cause.

**Bimodal by level.** On sibling levels, the runs that solved took a median of **12 min** (44 actions, 2.2 actions/min,
0.86K tokens/action); stuck runs had spent a median of 48 min, often ~100 (1.2 actions/min, 1.28K tokens/action).
Examples: tn36 L1 solved in 5 min vs 106/83 stuck; ft09 L4 8/5 vs 74; s5i5 L1 8/8 vs 85; su15 L1 4/9 vs 80.

**Solve rate per token (solves per 100K generated tokens):**

| condition | rate |
|---|---|
| fresh sprint at level entry (A0), sibling / all | 0.47 / 0.29 |
| goal-hinted sprint (A1), sibling | 0.79 (A1/A0 ratio 1.69, 90% 1.04–2.81) |
| route sprint (A2) | 6.3 |
| in-run, after 20K tokens unsolved, sibling / all | 1.11 / 0.80 |
| in-run, after 60K tokens unsolved, sibling / all | 0.69 / 0.47 |

All 25 games × 3 runs, every level reached, by tokens already spent on the level: 2.2 (0–10K), **3.6** (10–20K),
2.9 (20–40K), 1.9 (40–60K), 1.6 (60–100K), 1.1 (>100K). Tokens past 40K on a level are 31% of all tokens and 21% of
solves.

**What this rules out**
- *Fresh-context restart of a stuck level:* a fresh start (A0) solves at a lower rate per token than continuing, even
  after 60K stuck tokens. Agrees with `context-reset-does-not-pay`.
- *Goal-hint-style judgment as the big lever in real play:* in-run context on sibling levels (1.1) already beats the
  goal-hinted sprint (0.79).
- *Route search / closure tools:* the model already writes them; A2 shows execution is solved once the rules are.
- *Hazard-aware scheduling as a new idea:* dfranzen's priority gate already deprioritises games by tokens on the
  current level (chance proxy 0.5^((tokens/62K)^2)). The empirical decline is gentler than the proxy (≈ 0.3× at
  100K vs the proxy's 0.16×), so retuning `token_scale` is a small knob.

**What remains.** The binding quantity is the rate of correct rule induction per token, and it decays slowly rather
than locking. The large levers left are model-side (better rule induction per token) or throughput (more tokens
per level). Harness-side, nothing read here supports a ≥ 20% bet.
