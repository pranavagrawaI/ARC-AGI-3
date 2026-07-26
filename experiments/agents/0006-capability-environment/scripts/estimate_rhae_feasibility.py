"""Estimate RHAE population geometry, paired sensitivity, and cost for a 0006 comparison.

Two disjoint evidence bases are read:

1. ``analysis/article2-main-runs/shared/data`` — 400 archived game runs (16 GPT-5.4/5.5
   harness configurations x 25 games) with reconstructed official RHAE. This is the only
   source with completed levels, so it supplies the outcome distribution, floor/ceiling
   geometry, matched-pair variance, and environment action budgets.
2. ``experiments/agents/0002-llm-repl-agent/artifacts/runs`` — the local admissible-run
   manifests for the frozen 0002 substrate. This is the only source with the real core
   model, model-call pattern, token profile, and provider-reported cost. It is gitignored,
   so its derived table is written to ``data/`` for the committed record.

Outputs (written to ``experiments/agents/0006-capability-environment/data``):

* ``rhae-population-geometry.csv``   — per-configuration floor/ceiling mass
* ``rhae-paired-contrasts.csv``      — paired-difference variance for every matched contrast
* ``rhae-sensitivity.csv``           — minimum detectable effect vs. number of matched rounds
* ``local-run-cost-profile.csv``     — per-run admissible action/call/token/cost profile

Run:

    UV_CACHE_DIR=/tmp/uv-cache uv run python \
      experiments/agents/0006-capability-environment/scripts/estimate_rhae_feasibility.py
"""

from __future__ import annotations

import ast
import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
SHARED = REPO / "analysis" / "article2-main-runs" / "shared" / "data"
LOCAL_RUNS = REPO / "experiments" / "agents" / "0002-llm-repl-agent" / "artifacts" / "runs"
OUT = Path(__file__).resolve().parent.parent / "data"

RNG = np.random.default_rng(20260726)
SIMS = 40_000


# --------------------------------------------------------------------------- loading


@dataclass(frozen=True)
class GameRun:
    directory: str
    variant: str
    model: str
    effort: str
    game: str
    rhae: float
    completion_fraction: float
    levels_completed: int
    win_levels: int
    actions: int

    @property
    def cell(self) -> str:
        return f"{self.variant}|{self.model}|{self.effort}"

    @property
    def stratum(self) -> str:
        return f"{self.model}|{self.effort}"


def load_game_runs() -> list[GameRun]:
    rows: list[GameRun] = []
    with (SHARED / "per_game_outcomes.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                GameRun(
                    directory=row["Directory"],
                    variant=row["Variant"],
                    model=row["Model"],
                    effort=row["Effort"],
                    game=row["Game"],
                    rhae=float(row["RHAEScore"]),
                    completion_fraction=float(row["CompletionFraction"]),
                    levels_completed=int(row["LevelsCompleted"]),
                    win_levels=int(row["WinLevels"]),
                    actions=int(row["GlobalStep"]),
                )
            )
    return rows


def load_human_baselines() -> dict[str, list[int]]:
    baselines: dict[str, list[int]] = {}
    with (SHARED / "human_baselines.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            baselines[row["Game"]] = ast.literal_eval(row["BaselineActions"])
    return baselines


def load_local_manifests() -> list[dict[str, object]]:
    if not LOCAL_RUNS.is_dir():
        return []
    profile: list[dict[str, object]] = []
    for path in sorted(LOCAL_RUNS.glob("*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        usage = manifest.get("usage") or {}
        coach = manifest.get("human_coach") or {}
        score = manifest.get("host_score_components") or {}
        environments = score.get("environments") or []
        levels = sum(int(env.get("levels_completed") or 0) for env in environments)
        profile.append(
            {
                "run": path.parent.name,
                "model": manifest.get("model"),
                "game": manifest.get("game"),
                "human_coach_mode": coach.get("mode"),
                "actions": manifest.get("actions"),
                "max_actions": manifest.get("max_actions"),
                "model_calls": manifest.get("model_calls"),
                "python_calls": manifest.get("python_calls"),
                "levels_completed": levels,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get(
                    "reasoning_tokens"
                ),
                "cost_usd": usage.get("cost"),
                "stop_reason": manifest.get("stop_reason"),
                "state": manifest.get("state"),
            }
        )
    return profile


# ------------------------------------------------------------------- population shape


def population_geometry(runs: list[GameRun]) -> list[dict[str, object]]:
    cells: dict[str, list[GameRun]] = {}
    for run in runs:
        cells.setdefault(run.cell, []).append(run)

    rows: list[dict[str, object]] = []
    for cell, group in cells.items():
        scores = np.array([r.rhae for r in group])
        rows.append(
            {
                "cell": cell,
                "games": len(group),
                "mean_rhae": round(float(scores.mean()), 6),
                "median_rhae": round(float(np.median(scores)), 6),
                "sd_rhae": round(float(scores.std(ddof=1)), 6),
                "exact_zero": int((scores == 0).sum()),
                "below_1pct": int((scores < 0.01).sum()),
                "below_5pct": int((scores < 0.05).sum()),
                "at_or_above_99pct": int((scores >= 0.99).sum()),
                "wins": sum(1 for r in group if r.levels_completed == r.win_levels),
                "total_actions": sum(r.actions for r in group),
            }
        )
    rows.sort(key=lambda row: row["mean_rhae"])
    return rows


def floor_regression(rows: list[dict[str, object]]) -> tuple[float, float]:
    """Least-squares fit of below-1% count on configuration mean RHAE."""
    x = np.array([row["mean_rhae"] for row in rows])
    y = np.array([row["below_1pct"] / row["games"] for row in rows])
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


# ------------------------------------------------------------------- paired contrasts


def paired_contrasts(runs: list[GameRun]) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    by_cell: dict[str, dict[str, float]] = {}
    for run in runs:
        by_cell.setdefault(run.cell, {})[run.game] = run.rhae

    cells = sorted(by_cell)
    rows: list[dict[str, object]] = []
    vectors: dict[str, np.ndarray] = {}
    for i, left in enumerate(cells):
        for right in cells[i + 1 :]:
            l_variant, l_model, l_effort = left.split("|")
            r_variant, r_model, r_effort = right.split("|")
            if (l_model, l_effort) != (r_model, r_effort):
                continue  # only treatment contrasts inside a fixed model/effort stratum
            games = sorted(set(by_cell[left]) & set(by_cell[right]))
            a = np.array([by_cell[left][g] for g in games])
            b = np.array([by_cell[right][g] for g in games])
            diff = b - a
            key = f"{l_variant}->{r_variant}@{l_model}/{l_effort}"
            vectors[key] = diff
            rows.append(
                {
                    "contrast": key,
                    "pairs": len(games),
                    "mean_diff": round(float(diff.mean()), 6),
                    "sd_diff": round(float(diff.std(ddof=1)), 6),
                    "median_abs_diff": round(float(np.median(np.abs(diff))), 6),
                    "arm_corr": round(float(np.corrcoef(a, b)[0, 1]), 6),
                    "sd_left": round(float(a.std(ddof=1)), 6),
                    "sd_right": round(float(b.std(ddof=1)), 6),
                    "both_zero_pairs": int(((a == 0) & (b == 0)).sum()),
                    "sign_flips": int((np.sign(diff) != 0).sum()),
                }
            )
    rows.sort(key=lambda row: row["sd_diff"])
    return rows, vectors


def additive_residual_sd(runs: list[GameRun]) -> float:
    """Residual SD of an additive game + variant + model/effort fit.

    With one run per cell this cannot separate run-to-run noise from
    game x treatment interaction; it is a lower bound on the noise SD.
    """
    games = sorted({r.game for r in runs})
    variants = sorted({r.variant for r in runs})
    strata = sorted({r.stratum for r in runs})
    index = {}
    columns = 1 + (len(games) - 1) + (len(variants) - 1) + (len(strata) - 1)
    design = np.zeros((len(runs), columns))
    target = np.zeros(len(runs))
    for col, name in enumerate(
        [("game", g) for g in games[1:]] + [("variant", v) for v in variants[1:]]
        + [("stratum", s) for s in strata[1:]]
    ):
        index[name] = col + 1
    for row_i, run in enumerate(runs):
        design[row_i, 0] = 1.0
        for name in (("game", run.game), ("variant", run.variant), ("stratum", run.stratum)):
            col = index.get(name)
            if col is not None:
                design[row_i, col] = 1.0
        target[row_i] = run.rhae
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ beta
    dof = len(runs) - np.linalg.matrix_rank(design)
    return float(np.sqrt((residual**2).sum() / dof))


# ------------------------------------------------------------------------ sensitivity


def paired_t_mde(sd_diff: float, n_pairs: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """Minimum detectable paired mean difference, by Monte Carlo on the t statistic.

    The null critical value and the power are both simulated, so no t quantile
    function is needed and the same routine serves the permutation variant.
    """
    null = RNG.standard_normal((SIMS, n_pairs))
    stat = null.mean(axis=1) / (null.std(axis=1, ddof=1) / np.sqrt(n_pairs))
    crit = float(np.quantile(np.abs(stat), 1 - alpha))
    # under the alternative the same standardized statistic shifts by delta/sd*sqrt(n)
    lo, hi = 0.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        alt = null + mid / sd_diff
        alt_stat = alt.mean(axis=1) / (alt.std(axis=1, ddof=1) / np.sqrt(n_pairs))
        achieved = float((np.abs(alt_stat) > crit).mean())
        if achieved < power:
            lo = mid
        else:
            hi = mid
    return hi  # `mid` is already a raw mean difference, not a standardized one


def empirical_power(diff: np.ndarray, delta: float, alpha: float = 0.05, reps: int = 4000) -> float:
    """Sign-flip permutation power against a shift, resampling the observed contrast."""
    n = len(diff)
    centered = diff - diff.mean()
    hits = 0
    for _ in range(reps):
        sample = RNG.choice(centered, size=n, replace=True) + delta
        observed = sample.mean()
        signs = RNG.choice([-1.0, 1.0], size=(600, n))
        null = (signs * sample).mean(axis=1)
        pvalue = float((np.abs(null) >= abs(observed)).mean())
        hits += pvalue <= alpha
    return hits / reps


def discordant_pair_feasibility(
    q_values: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20, 0.40),
    needed: int = 6,
    target_power: float = 0.80,
) -> list[dict[str, object]]:
    """Rounds needed before a paired sign test can even reach two-sided p <= 0.05.

    Under a floored outcome every concordant-zero pair carries no information, so the
    binding constraint is the count of discordant pairs. Six same-direction discordant
    pairs is the smallest n with a two-sided sign-test p of 0.031.
    """
    rows: list[dict[str, object]] = []
    for q in q_values:
        rounds = None
        for candidate in range(1, 41):
            trials = RNG.binomial(25 * candidate, q, size=20_000)
            if float((trials >= needed).mean()) >= target_power:
                rounds = candidate
                break
        rows.append(
            {
                "per_game_discordance_rate": q,
                "expected_discordant_pairs_per_round": round(25 * q, 2),
                "rounds_for_6_discordant_at_80pct": rounds,
                "runs_per_arm": None if rounds is None else 25 * rounds,
            }
        )
    return rows


def first_level_cost(runs: list[GameRun]) -> dict[str, object]:
    """Actions the archived agents spent on level 1, the cheapest nonzero-RHAE event."""
    level_one: list[int] = []
    with (SHARED / "per_level_rhae.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["Level"] == "1" and row["Completed"] == "True":
                level_one.append(int(row["AgentActions"]))
    arr = np.array(level_one)
    return {
        "completed_level_1_records": int(len(arr)),
        "actions_median": float(np.median(arr)),
        "actions_p25": float(np.quantile(arr, 0.25)),
        "actions_p75": float(np.quantile(arr, 0.75)),
        "actions_p90": float(np.quantile(arr, 0.90)),
        "share_under_100_actions": round(float((arr <= 100).mean()), 4),
    }


# ---------------------------------------------------------------------------- budgets


def action_budgets(runs: list[GameRun], baselines: dict[str, list[int]]) -> dict[str, object]:
    actions = np.array([r.actions for r in runs])
    per_config = {}
    for run in runs:
        per_config.setdefault(run.cell, []).append(run.actions)
    sweep_totals = np.array([sum(v) for v in per_config.values()])
    human_total = sum(sum(v) for v in baselines.values())
    return {
        "per_game_mean": float(actions.mean()),
        "per_game_median": float(np.median(actions)),
        "per_game_p90": float(np.quantile(actions, 0.90)),
        "per_game_max": int(actions.max()),
        "sweep_total_mean": float(sweep_totals.mean()),
        "sweep_total_min": int(sweep_totals.min()),
        "sweep_total_max": int(sweep_totals.max()),
        "human_baseline_total": human_total,
    }


def run_reliability(profile: list[dict[str, object]]) -> dict[str, object]:
    """Transport/runtime mortality of the unsupervised wave-1 admissible runs."""
    wave = [row for row in profile if str(row["run"]).startswith("gemma-w1r-")]
    errored = [row for row in wave if row["stop_reason"] == "error"]
    return {
        "wave1_runs": len(wave),
        "ended_in_error": len(errored),
        "error_rate": round(len(errored) / len(wave), 4) if wave else None,
        "median_actions_before_error": (
            statistics.median([int(row["actions"]) for row in errored]) if errored else None
        ),
        "clean_terminations": [
            {"run": row["run"], "actions": row["actions"], "stop": row["stop_reason"]}
            for row in wave
            if row["stop_reason"] != "error"
        ],
    }


def wall_clock(profile: list[dict[str, object]], manifests_dir: Path) -> dict[str, object]:
    import datetime as dt

    seconds_per_action: list[float] = []
    for path in sorted(manifests_dir.glob("*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        coach = manifest.get("human_coach") or {}
        if coach.get("mode") == "interactive":
            continue  # human-in-the-loop latency is not machine throughput
        started, finished = manifest.get("started_at"), manifest.get("finished_at")
        actions = manifest.get("actions")
        if not (started and finished and isinstance(actions, int) and actions >= 20):
            continue
        elapsed = (dt.datetime.fromisoformat(finished) - dt.datetime.fromisoformat(started)).total_seconds()
        seconds_per_action.append(elapsed / actions)
    if not seconds_per_action:
        return {}
    return {
        "runs_used": len(seconds_per_action),
        "seconds_per_action_median": round(statistics.median(seconds_per_action), 1),
        "seconds_per_action_min": round(min(seconds_per_action), 1),
        "seconds_per_action_max": round(max(seconds_per_action), 1),
    }


def local_cost_profile(profile: list[dict[str, object]]) -> dict[str, object]:
    priced = [
        row
        for row in profile
        if isinstance(row.get("cost_usd"), (int, float))
        and isinstance(row.get("actions"), int)
        and row["actions"] >= 20
    ]
    if not priced:
        return {}
    cost_per_action = [row["cost_usd"] / row["actions"] for row in priced]
    calls_per_action = [
        row["model_calls"] / row["actions"] for row in priced if row.get("model_calls")
    ]
    prompt_per_call = [
        row["prompt_tokens"] / row["model_calls"] for row in priced if row.get("model_calls")
    ]
    completion_per_call = [
        row["completion_tokens"] / row["model_calls"] for row in priced if row.get("model_calls")
    ]
    prompt_per_action = [row["prompt_tokens"] / row["actions"] for row in priced]
    completion_per_action = [row["completion_tokens"] / row["actions"] for row in priced]
    return {
        "runs_used": len(priced),
        "longest_run_actions": max(row["actions"] for row in priced),
        "cost_per_action_median": statistics.median(cost_per_action),
        "cost_per_action_max": max(cost_per_action),
        "calls_per_action_median": statistics.median(calls_per_action),
        "prompt_tokens_per_call_median": statistics.median(prompt_per_call),
        "completion_tokens_per_call_median": statistics.median(completion_per_call),
        "prompt_tokens_per_action_median": statistics.median(prompt_per_action),
        "completion_tokens_per_action_median": statistics.median(completion_per_action),
        "admissible_levels_completed": sum(
            int(row["levels_completed"])
            for row in profile
            if row.get("human_coach_mode") in (None, "script")
        ),
    }


def qwen_cost(prompt_per_action: float, completion_per_action: float, actions: int) -> float:
    """Alibaba Model Studio list price for qwen3.6-27b: $0.60/M in, $3.60/M out."""
    return actions * (prompt_per_action * 0.60 + completion_per_action * 3.60) / 1e6


# ------------------------------------------------------------------------------- main


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    runs = load_game_runs()
    baselines = load_human_baselines()
    local = load_local_manifests()

    geometry = population_geometry(runs)
    write_csv(OUT / "rhae-population-geometry.csv", geometry)
    slope, intercept = floor_regression(geometry)

    contrasts, vectors = paired_contrasts(runs)
    write_csv(OUT / "rhae-paired-contrasts.csv", contrasts)
    residual_sd = additive_residual_sd(runs)

    sd_values = np.array([row["sd_diff"] for row in contrasts])
    sd_median = float(np.median(sd_values))
    sd_min = float(sd_values.min())
    noise_floor_sd = float(np.sqrt(2) * residual_sd)

    sensitivity_rows: list[dict[str, object]] = []
    for rounds in (1, 2, 3, 4, 6, 8):
        n_pairs = 25 * rounds
        for label, sd in (
            ("median_observed_contrast", sd_median),
            ("smallest_observed_contrast", sd_min),
            ("additive_residual_lower_bound", noise_floor_sd),
        ):
            # averaging k runs per arm per game shrinks the pair SD by sqrt(k)
            averaged_sd = sd / np.sqrt(rounds)
            sensitivity_rows.append(
                {
                    "rounds_per_arm": rounds,
                    "sd_basis": label,
                    "sd_diff": round(sd, 6),
                    "mde_pairs_25_averaged": round(paired_t_mde(averaged_sd, 25), 6),
                    "mde_pairs_pooled": round(paired_t_mde(sd, n_pairs), 6),
                    "runs_per_arm": 25 * rounds,
                }
            )
    write_csv(OUT / "rhae-sensitivity.csv", sensitivity_rows)

    if local:
        write_csv(OUT / "local-run-cost-profile.csv", local)
    cost = local_cost_profile(local)
    budgets = action_budgets(runs, baselines)

    representative_key = next(
        (key for key in vectors if key.startswith("ewma_v1.2_sv->twma_v1.2@5.5/xhigh")), None
    )
    representative = vectors.get(representative_key) if representative_key else None
    power_curve = {}
    if representative is not None:
        for delta in (0.05, 0.10, 0.15, 0.20, 0.30):
            power_curve[delta] = round(empirical_power(representative, delta, reps=800), 3)

    discordance = discordant_pair_feasibility()
    write_csv(OUT / "rhae-discordant-pair-feasibility.csv", discordance)

    summary = {
        "population": {
            "runs": len(runs),
            "games": len({r.game for r in runs}),
            "cells": len({r.cell for r in runs}),
            "weakest_cell": geometry[0],
            "strongest_cell": geometry[-1],
            "floor_fit_below_1pct_at_mean_rhae_0": round(intercept, 4),
            "floor_fit_slope": round(slope, 4),
        },
        "variance": {
            "sd_diff_median": round(sd_median, 6),
            "sd_diff_min": round(sd_min, 6),
            "sd_diff_max": round(float(sd_values.max()), 6),
            "additive_residual_sd": round(residual_sd, 6),
            "implied_noise_floor_pair_sd": round(noise_floor_sd, 6),
            "arm_corr_median": round(
                float(np.median([row["arm_corr"] for row in contrasts])), 4
            ),
        },
        "sensitivity": {
            "mde_25_pairs_median_sd": round(paired_t_mde(sd_median, 25), 6),
            "mde_25_pairs_noise_floor": round(paired_t_mde(noise_floor_sd, 25), 6),
            "representative_contrast": representative_key,
            "empirical_power_by_shift": power_curve,
            "discordant_pair_feasibility": discordance,
        },
        "first_level_cost": first_level_cost(runs),
        "budgets": budgets,
        "local_cost": cost,
    }
    summary["reliability"] = run_reliability(local)
    clock = wall_clock(local, LOCAL_RUNS)
    summary["throughput"] = clock

    if cost:
        scenarios = {
            # both arms, 25 games, one matched round
            "floor_probe_150_action_cap": 2 * 25 * 150,
            "gameover_regime_observed_locally": 2 * 25 * 60,
            "archive_regime_full_play": 2 * int(budgets["per_game_mean"] * 25),
        }
        spend: dict[str, object] = {}
        for name, actions in scenarios.items():
            entry = {
                "actions_total": actions,
                "model_calls_total": int(actions * cost["calls_per_action_median"]),
                "prompt_tokens_total": int(actions * cost["prompt_tokens_per_action_median"]),
                "completion_tokens_total": int(
                    actions * cost["completion_tokens_per_action_median"]
                ),
                "usd_observed_openrouter_rate": round(
                    cost["cost_per_action_median"] * actions, 2
                ),
                "usd_qwen_list_price": round(
                    qwen_cost(
                        cost["prompt_tokens_per_action_median"],
                        cost["completion_tokens_per_action_median"],
                        actions,
                    ),
                    2,
                ),
            }
            if clock:
                serial_hours = actions * clock["seconds_per_action_median"] / 3600
                entry["serial_hours"] = round(serial_hours, 1)
                entry["hours_at_5x_parallel"] = round(serial_hours / 5, 1)
            spend[name] = entry
        summary["spend_scenarios"] = spend

    print(json.dumps(summary, indent=2, default=float))
    (OUT / "rhae-feasibility-summary.json").write_text(
        json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
