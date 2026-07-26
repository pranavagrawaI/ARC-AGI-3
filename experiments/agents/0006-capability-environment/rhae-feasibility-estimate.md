# RHAE population, sensitivity, and cost estimate for a 0006 package comparison

Resolves [Estimate the feasible RHAE population, sensitivity, and cost from existing
evidence](https://github.com/pranavagrawaI/ARC-AGI-3/issues/5) on the map
[Wayfind the 0006 v0 capability-environment specification](https://github.com/pranavagrawaI/ARC-AGI-3/issues/1).

Produced by
[`scripts/estimate_rhae_feasibility.py`](./scripts/estimate_rhae_feasibility.py); every
number below is reproducible from the tables in [`data/`](./data/).

This estimate does **not** select a practical-effect threshold, a benefit or harm rule, an
exclusion rule, or a terminal-score convention. Those belong to
[Freeze the paired RHAE evaluation and practical-effect threshold](https://github.com/pranavagrawaI/ARC-AGI-3/issues/14).

## Two evidence bases, neither sufficient alone

| | Article 2 main-run archive | Local admissible 0002/0005 archive |
|---|---|---|
| Unit | 400 game runs = 16 configurations x 25 games | 58 run manifests |
| Core model | GPT-5.4 / GPT-5.5, high and xhigh | `qwen3.6-27b`, `google/gemma-4-31b-it` |
| Harness | EWMA / TWMA variants, not this repo's | 0002 generic LLM + Python workspace |
| Levels completed | 2,928 level records, 392 completed level 1s | **0 admissible level completions** |
| Provides | RHAE distribution, floor/ceiling geometry, paired variance, action budgets | model calls, token profile, provider cost, wall-clock, run mortality |
| Does not provide | anything about the frozen core model or this substrate | any nonzero RHAE observation |

The two do not overlap. The archive is the only source of RHAE variance; the local runs are
the only source of cost. Every combined figure below therefore rests on the assumption that
0006 and 0002-R will produce archive-shaped RHAE at local-archive cost per action. Section
["The floor is the decisive variable"](#the-floor-is-the-decisive-variable) argues that
assumption is currently false, and what would establish it.

## Population geometry and floor effects

Source: [`data/rhae-population-geometry.csv`](./data/rhae-population-geometry.csv).

The 25 archived games are the frozen population. Across all 16 configurations:

| Configuration | Mean RHAE | Median | SD | Exact zeros | < 1% | >= 99% | Wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ewma_v1.2_s` 5.4 high (weakest) | 30.59% | 17.38% | 0.327 | 2 | 3 | 2 | 5 |
| `ewma_v1.2` 5.5 high (median cell) | 51.14% | 50.14% | 0.391 | 0 | 0 | 6 | 12 |
| `ewma_v1.2_sv` 5.5 xhigh (strongest) | 74.78% | 75.13% | 0.247 | 0 | 0 | 7 | 20 |

Three geometric facts matter for the 0006 design:

1. **Per-game RHAE is bimodal, not bell-shaped.** Mass piles at 0 (unsolved level 1) and at
   the 1.15-per-level cap. Within-configuration SD across games (0.25–0.42) is comparable to
   the mean itself. Parametric normality assumptions are unsafe on a single round.
2. **The floor recedes with capability.** Sub-1% games fall from 3/25 at the weakest cell to
   0/25 at the four strongest. A least-squares fit gives `P(RHAE < 1%) ~ 0.244 - 0.364 x mean_RHAE`.
   **Do not use that intercept.** The archive contains no configuration below 30.6% mean
   RHAE, so the entire floor regime is extrapolation; the fit is reported only to show the
   direction and that the observed range never approaches the floor.
3. **Ceiling compression is real but secondary.** The strongest cell has 7/25 games at or
   above 99%, which caps the detectable upside of any treatment on those games.

## Matched units

The **game** is the pairing unit. Median across-arm correlation of per-game RHAE within a
model/effort stratum is **0.65**
([`data/rhae-paired-contrasts.csv`](./data/rhae-paired-contrasts.csv)), so pairing removes
roughly half the between-game variance and is clearly worth its cost. Level is available as a
finer unit (2,928 records) but levels within a game are not independent — later levels are
conditional on completing earlier ones, and RHAE's `i`-weighting and completion cap are
defined at game scope. Pairing below game scope would change the estimand.

Matched pairs available per round: **25**, fixed by the population. Additional statistical
information can only come from repeated rounds, not from more games.

## Variance and sensitivity

Every matched treatment contrast inside a fixed model/effort stratum (24 contrasts x 25
games) gives a paired-difference SD:

- median SD of paired differences: **0.283**
- range: **0.155 – 0.375**
- additive `game + variant + stratum` residual SD: 0.216, implying a pair-noise SD of
  `sqrt(2) x 0.216 = 0.305`

That last figure **exceeds** the median observed paired-difference SD. With one run per cell
the archive cannot separate treatment effect from run-to-run noise and game x treatment
interaction, so 0.283 should be read as a total paired-difference SD, not as a noise estimate
that a better design could shrink. It is nonetheless the right quantity for powering a paired
test.

Minimum detectable effect, paired test, alpha 0.05 two-sided, 80% power
([`data/rhae-sensitivity.csv`](./data/rhae-sensitivity.csv), Monte-Carlo on the t statistic,
40k draws):

| Rounds per arm | Runs per arm | MDE at SD 0.283 | MDE at SD 0.305 |
|---:|---:|---:|---:|
| 1 | 25 | **16.6 pp** | 17.8 pp |
| 2 | 50 | 11.7 pp | 12.7 pp |
| 4 | 100 | 8.3 pp | 8.9 pp |
| 8 | 200 | 5.8 pp | 6.3 pp |

A sign-flip permutation test resampled from the worst observed contrast agrees: power is
0.27 at a 10 pp shift, 0.49 at 15 pp, 0.73 at 20 pp.

**The decisive comparison.** All 24 observed harness-level treatment contrasts in the archive
have a mean RHAE difference below 14.5 pp; the median is **5.0 pp**. Every one of them is
smaller than the 16.6 pp single-round MDE. A one-round, 25-game paired comparison cannot
detect an effect of the size that harness changes actually produce in this benchmark. If the
0006 package effect is of ordinary harness magnitude, a single round is guaranteed to be
inconclusive, and 4–8 rounds per arm (100–200 runs per arm) is the honest sensitivity target.

## The floor is the decisive variable

The MDE table silently assumes both arms produce archive-shaped RHAE. The local archive says
they will not:

- **Zero admissible level completions.** Across all 58 local run manifests, no unsupervised
  admissible run has completed a single level of any game. The one level completion on record
  (`vc33`, 8 actions) is a human-coached run and inadmissible.
- **The failures are not action-budget failures.** Archived agents complete level 1 in a
  median of 19 actions, with 90.8% of the 392 completed level-1 records under 100 actions.
  The four clean local terminations reached 50–70 actions and ended in `GAME_OVER` with zero
  levels. The local agent had the actions and did not convert them.
- **RHAE is exactly zero for an agent that completes no level.** Both arms would score 0 on
  all 25 games, the paired difference vector would be identically zero, and no test — t,
  permutation, or sign — has any power. The endpoint is not merely underpowered; it is
  degenerate.

Under a floored outcome, statistical information is carried entirely by **discordant pairs**
(games where at least one arm scores nonzero and the arms differ). Six same-direction
discordant pairs is the smallest count reaching a two-sided sign-test p of 0.031. Rounds
needed to obtain six with 80% probability, as a function of the per-game discordance rate `q`
([`data/rhae-discordant-pair-feasibility.csv`](./data/rhae-discordant-pair-feasibility.csv)):

| `q` | Expected discordant pairs / round | Rounds needed | Runs per arm |
|---:|---:|---:|---:|
| 0.02 | 0.5 | 16 | 400 |
| 0.05 | 1.25 | 7 | 175 |
| 0.10 | 2.5 | 4 | 100 |
| 0.20 | 5.0 | 2 | 50 |
| 0.40 | 10.0 | 1 | 25 |

`q` is currently **unmeasured and best estimated at 0**. It is the single parameter that
decides whether official RHAE can function as the primary endpoint, and no existing artifact
constrains it.

## Action, model-call, and token budgets

**Environment actions** (archive, [`per_game_outcomes.csv`](../../../analysis/article2-main-runs/shared/data/per_game_outcomes.csv)):
mean 1,332 per game, median 1,517, p90 2,543, max 4,192. A 25-game sweep cost those
configurations 25,471–41,215 actions (mean 33,302). The official human baseline for all
levels of all 25 games totals 17,135 actions, so competitive play runs at roughly 2x human
action count.

**Model calls and tokens** (local admissible runs, 8 runs of >= 20 actions,
[`data/local-run-cost-profile.csv`](./data/local-run-cost-profile.csv)):

| Quantity | Median |
|---|---:|
| Model calls per environment action | 1.55 |
| Prompt tokens per model call | 16,823 |
| Completion tokens per model call | 571 |
| Prompt tokens per environment action | 24,948 |
| Completion tokens per environment action | 968 |

Per-action prompt cost is **flat in run length**, not growing: 24.4k tokens/action at 44
actions, 23.4k at 70, 19.5k at 93. This is the `max_context_turns = 6` rolling window doing
its job, and it is what licenses linear extrapolation from 93-action runs to 1,300-action
runs. If 0006's Context Assembler widens or unbounds the packet, this extrapolation breaks
and every cost figure below is a lower bound.

**Unmeasured 0006 overhead.** Nothing in the archive constrains how many extra model calls
the decision protocol, observable assertions, and reconciliation gate add per action. A
commitment turn plus a disposition turn on mismatch plausibly puts 0006 at 2–4x the 0002-R
call rate. Treat that multiplier as the largest single cost unknown; it also silently violates
the matched-budget assumption if budgets are specified in model calls rather than actions.

## Reliability and throughput

Of 13 unsupervised wave-1 admissible launches, **9 ended in a transport error**
(`ModelRequestError`, `IncompleteRead`), at a median of 8 actions in. Only 4 terminated
cleanly. That is a ~69% per-launch mortality rate, and under the frozen failure rules from
[Define the package-effect claim profile](https://github.com/pranavagrawaI/ARC-AGI-3/issues/4)
a mid-run provider failure after model exposure is not freely re-rollable. Any round budget
must carry a retry allowance, and the arm-independence of these failures must be
demonstrated, not assumed.

Wall-clock: **82.9 s per environment action** median across 6 unsupervised runs (range
28.2–118.0). This is throughput, not cost, and it is the tighter constraint.

## Spend and schedule scenarios

Both arms, 25 games, one matched round. `usd_observed` uses the measured Gemma-4-31b
OpenRouter rate of $0.004660/action; `usd_qwen_list` applies Alibaba Model Studio list pricing
for `qwen3.6-27b` ($0.60/M input, $3.60/M output) to the measured token profile.

| Scenario | Actions | Model calls | usd_observed | usd_qwen_list | Serial hours | Hours at 5x parallel |
|---|---:|---:|---:|---:|---:|---:|
| Floor probe, 150-action cap | 7,500 | 11,589 | $35 | $138 | 173 | 35 |
| `GAME_OVER` regime observed locally (~60 actions) | 3,000 | 4,635 | $14 | $55 | 69 | 14 |
| Archive regime, full play (~1,332 actions/game) | 66,604 | 102,923 | $310 | $1,229 | 1,534 | 307 |

Multiply the third row by the number of rounds sensitivity demands: 4 rounds is roughly
$4,900 and 6,100 serial hours (~51 days at 5x parallelism); 8 rounds is roughly $9,800.
A ~1.5x retry allowance for the observed mortality rate takes 4 rounds to ~$7,400; composing
that with a 2–4x 0006 call overhead takes it to **$15,000–$30,000 and several months of
wall-clock** at 5x parallelism.

The structural result: **the evaluation is cheap exactly when it is uninformative and
expensive exactly when it is informative.** A floored agent terminates in ~60 actions and
costs $14 a round while producing an identically-zero difference vector. An agent capable
enough to make RHAE non-degenerate must survive ~1,300 actions per game, and then a properly
powered comparison costs four figures per round and months of serial time.

## What this implies for downstream tickets

Reported as consequences, not as decisions — the choices belong to the tickets named.

1. **[#14, paired RHAE evaluation and threshold]** must confront a single round's 16.6 pp
   MDE against a 5.0 pp median observed harness effect, and must decide what to report when
   the difference vector is degenerate. "No significant difference" is not a licensed reading
   of an all-zero comparison; that is `UNIDENTIFIABLE` under the frozen outcome profile.
2. **A bounded floor probe is the cheapest way to buy the missing parameter.** A 150-action
   cap covers the 90th percentile of archived level-1 completion cost, costs $35–138 and ~35
   hours at 5x parallelism, and measures `q` directly. It selects no threshold and commits no
   design. It converts the central unknown into an observation before #14 freezes anything.
3. **Interface Readiness Gate and Local Prospective Consequence fixtures are cheap; RHAE is
   not.** All the fixture-bound components of the outcome profile are bounded-fixture work.
   Sequencing them first is already the frozen stage rule, and this estimate strengthens it:
   they cost ~1% of the live comparison.
4. **The 69% run mortality rate is a specification input**, not an operational detail. It
   belongs in
   [#15, audit artifacts, containment, and operational failure handling](https://github.com/pranavagrawaI/ARC-AGI-3/issues/15)
   as a retry-allowance and arm-independence-evidence requirement.

## What would change these numbers

- **A single admissible level completion by the frozen core model on this substrate.** It
  would put `q > 0` and move the whole analysis out of the degenerate regime.
- **A replicated cell.** Two runs of one configuration on all 25 games would separate
  run-to-run noise from game x treatment interaction and could materially lower the required
  round count. The archive's one-run-per-cell design makes this currently impossible.
- **A measured 0006 call multiplier.** Any readiness-fixture run that logs model calls per
  action under the full decision protocol replaces the 2–4x assumption with data.
- **A context policy that unbounds the packet.** This would invalidate the flat per-action
  token profile and every cost figure derived from it.

## Assumptions and limits

- Archive RHAE was reconstructed, not officially reported; the reconstruction was verified
  against the official `EnvironmentScoreCalculator` to ~5e-10 pp
  ([`rhae-overview.md`](../../../analysis/article2-main-runs/shared/rhae-overview.md)).
- Archived short game IDs were mapped to current versioned IDs by exact game-set and
  level-count agreement, not by encoded version identity.
- Cost per action is measured on `google/gemma-4-31b-it` via OpenRouter, not on the frozen
  core model; the Qwen figure is list price applied to a measured token profile, and neither
  includes prompt-caching effects that a production run would obtain.
- `data/local-run-cost-profile.csv` is committed because its source,
  `experiments/agents/0002-llm-repl-agent/artifacts/runs/`, is gitignored and cannot be cited
  directly.
- Wall-clock and mortality figures come from 2026-07-20 to 2026-07-23 provider conditions and
  are not a stable property of the design.
