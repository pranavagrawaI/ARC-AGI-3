<div align="center">

<img src=".github/figures/banner.png" width="92%" alt="TWIN: writes the game it plays. 93.3 on ARC-AGI-3">

**Playing an unknown game by building its digital twin at test time.**

A coding agent writes an executable copy of an [ARC-AGI-3](https://three.arcprize.org) game in Python,<br>
validates it for free against every observed transition, plans inside it,<br>
and spends scored actions only on verified plans or deliberate probes.

[![Site](https://img.shields.io/badge/site-arc--agi--3--twin-3E7DE8.svg)](https://arc-agi-3-twin.vercel.app)
[![Benchmark](https://img.shields.io/badge/benchmark-ARC--AGI--3-6d28d9.svg)](https://three.arcprize.org)
[![Python](https://img.shields.io/badge/python-3.12-3776ab.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

<br>

<img src=".github/figures/fig_loop_v4.png" width="92%" alt="The TWIN loop">

<sub><b>The loop, drawn with real states from the game ft09.</b> Everything left of the scored
boundary is free: writing the twin, replaying it over the whole log, diagnosing what blocks
progress, and searching for a route. Only <code>ExecuteChecked</code> submits real actions, one at a
time, halting the moment reality disagrees with the prediction.</sub>

<br><br>

<table>
<tr>
<td width="50%" align="center">
<img src=".github/figures/demo/sp80.gif" width="100%" alt="TWIN playing sp80"><br>
<sub><b>sp80</b> · the twin stays 92% exact while the agent climbs five levels</sub>
</td>
<td width="50%" align="center">
<img src=".github/figures/demo/primitives_dc22.gif" width="100%" alt="Discovering primitives on dc22"><br>
<sub><b>dc22</b> · 39 of 40 frames predicted exactly; validate catches the one miss</sub>
</td>
</tr>
</table>

<br>

### Results at a glance

| | Score (/100) | Games won | Levels cleared |
|---|:---:|:---:|:---:|
| **TWIN** (ours) | **93.3** | **23 / 25** | **179 / 183** |
| EWM | 63.8 | 14 / 25 | 146 / 183 |
| Plain Codex harness | 61.1 | 13 / 25 | 148 / 183 |
| Base model, played directly | 7.8 | 1 / 25 | — |

Same base model, three harnesses: **7.8 → 61.1 → 93.3**.

</div>

---

## The idea

**The asymmetry.** ARC-AGI-3 scores only the game actions you submit. Reasoning, file
I/O, and any code you run are free, and the score rewards clearing a game in as few
actions as a human playing it for the first time. That makes offline computation the
cheap resource and real actions the expensive one. TWIN spends the cheap resource until
it has something worth committing.

**The twin.** The agent's theory of the game is a Python file with a fixed contract:

```python
step(grid, action) -> grid          # the transition function
goal_reached(grid) -> bool          # the hidden win condition
```

It starts as an identity stub, so every rule in it comes from interaction. Because both
sides are raw 64x64 grids, the harness can check the twin cell by cell against every
logged transition. A submit gate refuses any scored action while that replay fails.
Each mismatch during execution returns as a counterexample and repairs the code.

**The asymmetric objective.** Fit and reachability are treated differently. Fit is a
hard precondition: the twin must reproduce *every* observed transition before a single
scored action issues. Reachability is only sought: the planner needs *one* imagined
route to the goal, and finding none simply sends control back to exploration. The win
condition is never stated, so TWIN proposes a reachable state as a tentative goal,
checks that no logged frame satisfies it, and buys the cheapest test of it. A level
boundary confirms the hypothesis; reaching the state without a boundary kills it.

---

## Results

All 25 public games, 183 levels, one run per game, no parameter search. The score is
the benchmark's action-efficiency metric: `min(1.15, (human/agent)^2)` per cleared
level, index-weighted, capped by completion. Only the Codex ablation shares TWIN's
base model, so it is the clean attribution: 32.2 points below.

<p align="center">
  <img src=".github/figures/fig_scoremap.png" width="88%" alt="Per-game scores">
</p>

Per-game scores, ordered by TWIN's score. TWIN alone clears bp35, lf52, and sk48, the
nine-, ten-, and eight-level games where the published systems stall early. Its two
failures are sc25 and sp80, where a single deep level makes every goal hypothesis
expensive to test.

<p align="center">
  <img src=".github/figures/fig_parity.png" width="88%" alt="Action efficiency against humans">
</p>

On the 23 games it wins, TWIN uses 0.61x as many actions as the human reference, and
only tn36 (1.5x) and ka59 (1.6x) sit above parity. Across the 13 games every compared
system fully clears, TWIN spends 3,357 scored actions against 5,381 for EWM and 7,485
for humans.

<details>
<summary><b>Full per-game results (all 25 games)</b></summary>

<br>

act = scored actions, lvl = levels cleared. Every system is scored against the same
published human baselines.

| Game | Human act | TWIN act | TWIN lvl | TWIN | EWM |
|---|---:|---:|:---:|---:|---:|
| `ar25` | 748 | 254 | 8/8 | **100.0** | 100.0 |
| `bp35` | 651 | 529 | 9/9 | **100.0** | 17.3 |
| `cd82` | 171 | 93 | 6/6 | **100.0** | 100.0 |
| `cn04` | 789 | 261 | 6/6 | **100.0** | 100.0 |
| `dc22` | 1,228 | 1,219 | 6/6 | **97.1** | 36.4 |
| `ft09` | 208 | 80 | 6/6 | **100.0** | 95.1 |
| `g50t` | 879 | 577 | 7/7 | **90.2** | 100.0 |
| `ka59` | 730 | 1,138 | 7/7 | **73.3** | 63.0 |
| `lf52` | 1,339 | 829 | 10/10 | **100.0** | 34.9 |
| `lp85` | 388 | 104 | 8/8 | **100.0** | 100.0 |
| `ls20` | 776 | 515 | 7/7 | **100.0** | 98.3 |
| `m0r0` | 1,107 | 236 | 6/6 | **100.0** | 71.6 |
| `r11l` | 233 | 85 | 6/6 | **100.0** | 74.3 |
| `re86` | 1,255 | 750 | 8/8 | **100.0** | 26.4 |
| `s5i5` | 638 | 348 | 8/8 | **100.0** | 0.7 |
| `sb26` | 213 | 137 | 8/8 | **100.0** | 100.0 |
| `sc25` | 350 | 701 | 3/6 | **32.7** | 20.1 |
| `sk48` | 1,070 | 752 | 8/8 | **87.2** | 30.2 |
| `sp80` | 518 | 1,037 | 5/6 | **82.1** | 5.5 |
| `su15` | 361 | 129 | 9/9 | **100.0** | 88.7 |
| `tn36` | 317 | 475 | 7/7 | **69.7** | 26.2 |
| `tr87` | 414 | 162 | 6/6 | **100.0** | 86.3 |
| `tu93` | 462 | 236 | 9/9 | **100.0** | 100.0 |
| `vc33` | 447 | 243 | 7/7 | **100.0** | 19.4 |
| `wa30` | 1,843 | 724 | 9/9 | **100.0** | 100.0 |
| **mean / total** | | | **179/183** | **93.3** | **63.8** |

</details>

**Contamination controls.** Web search is disabled; an integrity audit screens every
transcript and agent log for deny-listed reads. The games postdate the base model's
training cutoff, so every rule in a twin is learned inside the run.

---

## Does the world model actually learn the game?

Replay validation cannot distinguish a general model from a memorized one. The honest
test runs post hoc: at visited states, apply actions the agent never took, and compare
the final twin against the real engine cell by cell. No agent saw ground truth during
play.

<p align="center">
  <img src=".github/figures/fig_wm_accuracy.png" width="88%" alt="World-model accuracy per game">
</p>

Per game, open circles are the accuracy of the agent's precommitted predictions at
acting time, on the actions it took. Filled circles are the final twin on actions it
never took, and they order the games.

| Measurement | Tests | Exact frames | Cell match |
|---|---:|---:|---:|
| Final twin replays its own log | 11,314 | 99.4% | 0.9996 |
| Final twin vs engine, actions the agent took | 2,354 | 95.4% | 0.9981 |
| Final twin vs engine, **actions never taken** | 20,790 | 65.3% | 0.9950 |
| the same, excluding order-sensitive sp80 | 19,583 | **68.6%** | 0.9954 |
| level- or game-crossing probes | 98 | 12.2% | 0.9009 |

**Excluding order-sensitive sp80, the final twins predict 68.6% of never-taken actions
exactly.** The spread matters more than the mean: tu93's twin generalizes at 98.8%,
wa30's lookup-table twin replays its own log and knows almost nothing off it (9.9%),
and both score 100.0 — which is why the off-log number is worth measuring. The twins
also predict 79% of first-seen pairs exactly, and the first goal hypothesis is right
on 87.2% of cleared levels. Fitting the model is the tractable half; inferring the
goal is the frontier.

---

## What the agent actually does

<div align="center">

<table>
<tr>
<td align="center" width="33%"><img src=".github/figures/demo/ar25.gif" width="100%"><br><sub><b>ar25</b> · 254 actions, human 748</sub></td>
<td align="center" width="33%"><img src=".github/figures/demo/vc33.gif" width="100%"><br><sub><b>vc33</b> · 243 actions, human 447</sub></td>
<td align="center" width="33%"><img src=".github/figures/demo/primitives_vc33.gif" width="100%"><br><sub><b>vc33</b> · the twin's plan, probes, and imagined win beside reality</sub></td>
</tr>
</table>

<sub>Rendered from the run logs. Every frame is one real submitted action.</sub>

</div>


<p align="center">
  <img src=".github/figures/fig_actions.png" width="88%" alt="Every scored action">
</p>

Every scored action of all 25 runs, one row per game. Blue executes a validated plan,
amber is a deliberate probe, red is a mismatch that becomes a counterexample.
**Probes are 7.1% of scored actions and front-loaded; the other 92.9% execute plans
already tested inside the twin.**

<p align="center">
  <img src=".github/figures/fig_learning2.png" width="88%" alt="Online prediction accuracy">
</p>

The predicted next frame is committed before each plan action, then checked against
the frame the game returns. Per game the rate runs from 98% (lf52) to 29% (lp85);
low-accuracy games still win because the executor halts at the first mismatch and
repairs from the counterexample.

---

## Cost

<p align="center">
  <img src=".github/figures/fig_compute_pareto.png" width="88%" alt="Score versus tokens">
</p>

| System | Score | Levels | Tokens | Scored actions | Tokens/action |
|---|---:|---:|---:|---:|---:|
| TWIN | 93.3 | 179/183 | 2.60B | 11,614 | 224k |
| Ablation, same agent without the twin | 61.2 | 147/183 | 1.06B | 22,224 | 48k |
| EWM (published run) | 63.8 | 146/183 | 20.0B | 28,035 | 715k |

The ablation row is recomputed from the banked artifacts, which gives 61.2 and 147
levels; the results table above quotes the paper's frozen 61.1 and 148.

2.60B processed tokens over 91.4 hours. Against EWM: 7.7x fewer tokens for 29.5 more
points. Against its own ablation: 2.4x the tokens, half the scored actions, +32 points
— action efficiency is bought with compute. Cached reads are 98% of tokens, and spend
follows difficulty: ka59 alone burns 24% of the campaign.

---

## Quickstart

```bash
uv sync                       # once: install into .venv
uv run twin fetch-games       # once: game files for local play (no ARC key needed)

uv run twin play -p codex -g ft09          # TWIN via Codex CLI
uv run twin play -p claude-code -g ft09    # same method, Claude Code as the agent
```

`-p` selects the policy, `-g` the game id. Registered policies are `codex`,
`claude-code`, `explore`, and `randomwalk`. The two agent policies drive a coding-agent
CLI that must be installed and authenticated separately. `explore` and `randomwalk` need
no model access, so they are the quickest way to exercise the harness end to end.

Useful flags: `--max-steps N` overrides the default budget of twice the game's published
human baseline, and `--memory-from <run-dir>` seeds from a previous run's twin.

The ablation, same agent and sandbox with the harness disabled:

```bash
ARC_CLI_MINIMAL=1 ARC_CLI_BRIEF_FILE=twin/cliagent/plain_brief.md \
  TELL_RUNS_DIR=runs/plain uv run twin play -p codex -g ft09
```

Campaign helpers used for the 25-game run:

```bash
scripts/run/twin_guarded.sh <games...>            # stall guard: 40 min without a tick, kill and retry
uv run python scripts/run/fleet_status.py <roots> # progress across run roots
```

**What a run writes.** `runs/<campaign>/<game>/<timestamp>_<policy>_<model>/` holds
`result.json` (benchmark score plus the integrity verdict), `memory/model.py` (the
learned twin), `transitions.jsonl`, `transcript.jsonl`, `replay.html`, and
`trajectory.gif`. The integrity audit in `twin/integrity.py` scans both the transcript
and the raw agent log; a read of a ground-truth source in either invalidates the run.

<details>
<summary><b>Environment variables (all optional)</b></summary>

<br>

| Variable | Default | Purpose |
|---|---|---|
| `TELL_RUNS_DIR` | `runs/...` | where run directories are written |
| `ARC_CODEX_BIN` / `ARC_CLAUDE_BIN` | `codex` / `claude` | agent binaries |
| `ARC_CODEX_MODEL` / `ARC_CLAUDE_MODEL` | `''` / `fable` | model slug |
| `ARC_CLI_EFFORT` | `xhigh` | reasoning effort |
| `ARC_CLI_TURN_TIMEOUT` | `900` | seconds to wait for the agent's next action |
| `ARC_CLI_MAX_RESPAWNS` | `50` | CLI relaunches with a RESUME prompt |
| `ARC_CLI_MINIMAL` | unset | ablation: bare `play.py`, no world-model gate |
| `ARC_CLI_BRIEF_FILE` | unset | override the agent brief |

The Codex path needs `codex` on PATH and logged in; the model comes from your
`~/.codex/config.toml` pin. The spawn runs under a workspace-only, network-off sandbox
profile that is preflighted before play.

</details>

---

---

<details>
<summary><b>Limitations</b></summary>


- **Deterministic grid worlds.** Replay validation assumes dynamics the twin can
  represent deterministically, and certifies only the transitions actually logged. A
  probabilistic twin would loosen that boundary.
- **Fixed search budgets.** Planning runs at depth 8 with 20,000 nodes, goal search at
  depth 14 with 30,000. A goal beyond the horizon goes unfound.
- **One frame of context.** `step` reads a single grid, so mechanics driven by history
  the grid does not show are out of scope. Exact replay also presumes small discrete
  states. Cell equality on 64x64 grids is free; high-dimensional observations would need
  an approximate comparison with a threshold to tune.
- **One run per game.** Every number here comes from a single scored run of each of the
  25 games, so per-game variance is not measured. The hosted model exposes no seed, and
  reproducibility rests on the audit trail, where every scored action carries a
  prediction recorded before execution.
- **Two games unfinished.** sc25 (32.7) and sp80 (82.1) are goal failures, not dynamics
  failures. sp80's twin stays 92.3% accurate while the sixth level's win condition
  resists. Inferring the goal, not fitting the model, is the frontier.

</details>

## License

MIT. See [`LICENSE`](LICENSE).
