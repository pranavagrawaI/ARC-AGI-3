# Decoded: ls20's real win condition (and why navigation alone can't clear levels)

Read the obfuscated game source `environment_files/ls20/9607627b/ls20.py` rather than guessing the mechanic. Findings, with line refs:

- **Level clear** (`pbznecvnfr`, line 2042): for EVERY goal door (`plrpelhym`), the player must stand exactly on the door tile (x==x, y==y) AND satisfy `bejndxqqzf`.
- **The match gate** (`bejndxqqzf`, line 2040): `shape == required AND color == required AND rotation == required`. Per-level requirements come from level data `StartShape`/`GoalColor`/`GoalRotation`.
- **Attribute changers** (`txnfzvzetn`, lines 1893–1909): three tile tags CYCLE one attribute each per contact — shape (`ttfwljgohq`), color (`soyhouuebz`), rotation `%4` (`rhsxkxzdjz`). They cycle, not set — so reaching a target state can take multiple taps.
- **Wall** = tag `ihdgageizm` (blocks). **Door reached with wrong attrs** = blocked + penalty `akoadfsur` (red flash). **Movement**: ACTION1/2/3/4 = up/down/left/right, one cell = 5px (confirmed empirically: median deltas (-5,0)/(5,0)/(0,-5)/(0,5)).

**Curriculum correction:** lesson 0005's HTML had promised 0006 might "clear all seven levels." The source proves navigation alone NEVER clears a level (door gated on matching). So "competence" splits into two skills: (0006) spatial planning, (0007) the matching puzzle. The 0006 HTML now states this correction explicitly — a teaching beat about reading source vs guessing (Karpathy: know your data/environment cold).

**Lesson 0006 built + verified** (`teach/lessons/0006-systematic-explorer.py` + `0006-the-systematic-explorer.html`): movement graph + BFS to nearest frontier. New skill = BFS. Incorporated his spawn-poison fix as `observe`'s wall rule (`0 < changed < 100`, excluding the zero-change startup no-op). Measured live 2026-06-22, 5 ep: coverage@150 random ~25–33 vs systematic ~33–36 (full room); final coverage both 36 (random ~3× slower); levels ~0 both (rare fluke level 1). Key honest finding: planning's edge is SPEED not final coverage, and it scales with map size — marginal in a 36-cell room. Headline metric changed from final-coverage (didn't separate) to coverage@150 (does).

**Awaiting from him:** predictions (coverage@150, levels) + explanation of why coverage moved but levels didn't; bonus warm-up for 0007: fewest taps to cycle a 4-state rotation 0→3 (answer: 3, cyclers only go one direction).

**Next (lesson 0007):** the matching puzzle — detect each door's required shape/color/rotation and the player's current, compute the gap, BFS to each needed cycler and tap N times, BFS to the door. First agent expected to move the levels column.
