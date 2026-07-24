# Textual versus executable-model divergence corpus

## Question and comparison unit

This analysis looks for games where the **TWMA textual world-model scaffold** and the **EWMA-SV executable-model plus verification scaffold** differ strongly.

Each comparison holds fixed:

- game;
- underlying language model (`gpt-5.4` or `gpt-5.5`);
- reasoning effort (`high` or `xhigh`);
- run index (`run1`).

The target metric is official reconstructed ARC-AGI-3 RHAE, not win count or unweighted level completion. `ExecutableMinusTextPP` is EWMA-SV RHAE minus TWMA RHAE in percentage points.

Source: [`../../shared/data/per_game_outcomes.csv`](../../shared/data/per_game_outcomes.csv). The reconstruction represented there was independently checked against the official toolkit across all 400 runs, as documented in [`../../shared/rhae-overview.md`](../../shared/rhae-overview.md).

Generated evidence:

- [`data/text-vs-executable-pairs.csv`](./data/text-vs-executable-pairs.csv): all 100 matched game/model/effort cells;
- [`data/text-vs-executable-game-summary.csv`](./data/text-vs-executable-game-summary.csv): four-cell summary for each of 25 games;
- `scripts/rank_text_vs_executable.py`: reproducible ranking script.

## Recommended balanced trace corpus

Exclude `sc25` and `r11l`, which are already selected for the model-version comparison. Also treat `ka59` as already covered by the existing detailed audit.

The cleanest next corpus uses `xhigh` throughout and includes one result favoring each scaffold under each underlying model:

| Underlying model | Game | Textual TWMA | Executable EWMA-SV | SV − text | Outcome class |
|---|---|---:|---:|---:|---|
| GPT-5.4 xhigh | `g50t` | 5/7, 29.135% RHAE, 3,485 actions | 7/7, 93.883%, 531 actions | **+64.748 pp** | executable win flip |
| GPT-5.4 xhigh | `cn04` | 6/6, 77.573%, 643 actions | 2/6, 9.815%, 1,594 actions | **−67.758 pp** | textual win flip |
| GPT-5.5 xhigh | `bp35` | 6/9, 9.405%, 2,365 actions | 9/9, 66.883%, 985 actions | **+57.477 pp** | executable win flip |
| GPT-5.5 xhigh | `vc33` | 7/7, 69.063%, 588 actions | 3/7, 20.552%, 1,652 actions | **−48.512 pp** | textual win flip |

This is more discriminating than taking the four largest absolute cells. It prevents the corpus from selecting only executable-scaffold successes and tests whether a proposed mechanism explains both benefit and harm while model and effort remain fixed.

The archived `agent parameters` records confirm the advertised model and `xhigh` effort for all eight selected runs. Their complete directories contain 197,579 files and 717,353,715 logical bytes. Copying should therefore occur only after accepting this four-game scope, not as part of an indiscriminate archive migration.

## Other large divergences

### Executable scaffold favored

Excluding `sc25`, `r11l`, and already-audited `ka59`:

| Game/cell | SV − text | Outcome |
|---|---:|---|
| `g50t`, GPT-5.4 xhigh | +64.748 pp | executable win flip |
| `sb26`, GPT-5.4 xhigh | +62.615 pp | executable win flip |
| `re86`, GPT-5.5 high | +59.138 pp | executable win flip |
| `bp35`, GPT-5.5 xhigh | +57.477 pp | executable win flip |
| `wa30`, GPT-5.5 xhigh | +57.326 pp | executable win flip |
| `s5i5`, GPT-5.4 xhigh | +54.110 pp | neither solved |
| `ft09`, GPT-5.5 high | +52.381 pp | executable win flip |
| `sk48`, GPT-5.5 xhigh | +48.500 pp | executable win flip |

### Textual scaffold favored

| Game/cell | SV − text | Outcome |
|---|---:|---|
| `tn36`, GPT-5.5 high | −75.070 pp | textual win flip |
| `cn04`, GPT-5.4 xhigh | −67.758 pp | textual win flip |
| `cn04`, GPT-5.5 high | −52.381 pp | textual win flip |
| `vc33`, GPT-5.5 xhigh | −48.512 pp | textual win flip |
| `vc33`, GPT-5.5 high | −46.975 pp | textual win flip |
| `dc22`, GPT-5.5 xhigh | −44.699 pp | textual win flip |
| `cn04`, GPT-5.5 xhigh | −32.456 pp | both solved; efficiency difference |
| `ft09`, GPT-5.5 xhigh | −30.825 pp | both solved; efficiency difference |

## Cross-cell stability

A large single-cell difference can be a useful trace discriminator without being a stable treatment effect. Across the four model/effort cells per game:

- `bp35`: executable scaffold scores higher in **4/4** cells; mean difference +26.840 pp.
- `vc33`: textual scaffold scores higher in **4/4** cells; mean difference −31.370 pp.
- `g50t`: executable scaffold scores higher in **3/4** cells; mean difference +33.815 pp.
- `cn04`: textual scaffold scores higher in **3/4** cells; mean difference −27.605 pp.

These are descriptive consistencies over one run per cell, not replicated causal estimates.

## What trace analysis must distinguish

For each selected pair, the audit should not infer causality from the score gap alone. It should test:

1. whether EWMA-SV actually instantiated executable modeling and verification rather than merely being prompted to do so;
2. where charged actions went: discriminating probes, repeated probes, failed plans, resets, stale batch tails, and recovery;
3. whether verifier failures changed subsequent behavior before more charged actions;
4. whether the textual run succeeded through a better world model, a shorter planning path, lower construction overhead, or luck in an unresolved mechanic;
5. whether the same candidate mechanism predicts both the positive and negative games.

One run per arm supports adversarial mechanism diagnosis, not an average treatment-effect claim.
