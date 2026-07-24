#!/usr/bin/env python3
"""Rank matched textual-vs-executable ARC-AGI-3 runs by official RHAE.

The input is the verified per-game outcome index. One output row compares the
TWMA textual scaffold with EWMA-SV executable verification while holding game,
model, effort, and run fixed. A second output aggregates those four matched
cells per game without treating them as replicated causal estimates.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

TEXT_VARIANT = "twma_v1.2"
EXEC_VARIANT = "ewma_v1.2_sv"
PAIR_FIELDS = [
    "Game",
    "Model",
    "Effort",
    "Run",
    "TextDirectory",
    "ExecutableDirectory",
    "TextState",
    "ExecutableState",
    "TextVerifiedSolved",
    "ExecutableVerifiedSolved",
    "TextLevelsCompleted",
    "ExecutableLevelsCompleted",
    "WinLevels",
    "TextActions",
    "ExecutableActions",
    "TextRHAEPercent",
    "ExecutableRHAEPercent",
    "ExecutableMinusTextPP",
    "AbsoluteDifferencePP",
    "OutcomeClass",
]
SUMMARY_FIELDS = [
    "Game",
    "MatchedCells",
    "MeanExecutableMinusTextPP",
    "MeanAbsoluteDifferencePP",
    "MaxAbsoluteDifferencePP",
    "MaxDifferenceModel",
    "MaxDifferenceEffort",
    "MaxDifferenceSignedPP",
    "MaxDifferenceOutcomeClass",
    "ExecutableBetterCells",
    "TextBetterCells",
    "TiedCells",
    "ExecutableWinFlipCells",
    "TextWinFlipCells",
    "BothSolvedCells",
]


def parse_args() -> argparse.Namespace:
    comparison_root = Path(__file__).resolve().parents[1]
    analysis_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=analysis_root / "shared/data/per_game_outcomes.csv",
    )
    parser.add_argument(
        "--pairs-output",
        type=Path,
        default=comparison_root / "data/text-vs-executable-pairs.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=comparison_root / "data/text-vs-executable-game-summary.csv",
    )
    return parser.parse_args()


def solved(row: dict[str, str]) -> bool:
    value = row["VerifiedSolved"].strip().lower()
    if value not in {"true", "false"}:
        raise ValueError(f"invalid VerifiedSolved value: {row['VerifiedSolved']!r}")
    return value == "true"


def outcome_class(text: dict[str, str], executable: dict[str, str]) -> str:
    text_solved = solved(text)
    executable_solved = solved(executable)
    if executable_solved and not text_solved:
        return "executable_win_flip"
    if text_solved and not executable_solved:
        return "text_win_flip"
    if text_solved and executable_solved:
        return "both_solved"
    return "neither_solved"


def write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    with args.input.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))

    relevant = [row for row in source_rows if row["Variant"] in {TEXT_VARIANT, EXEC_VARIANT}]
    by_key: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
    for row in relevant:
        key = (row["Variant"], row["Model"], row["Effort"], row["Run"], row["Game"])
        if key in by_key:
            raise AssertionError(f"duplicate configuration/game row: {key}")
        by_key[key] = row

    models = sorted({row["Model"] for row in relevant})
    efforts = sorted({row["Effort"] for row in relevant})
    runs = sorted({row["Run"] for row in relevant})
    games = sorted({row["Game"] for row in relevant})

    pairs: list[dict[str, Any]] = []
    for model in models:
        for effort in efforts:
            for run in runs:
                for game in games:
                    text_key = (TEXT_VARIANT, model, effort, run, game)
                    exec_key = (EXEC_VARIANT, model, effort, run, game)
                    if text_key not in by_key or exec_key not in by_key:
                        raise AssertionError(f"missing matched treatment for {model}/{effort}/run{run}/{game}")
                    text = by_key[text_key]
                    executable = by_key[exec_key]
                    if text["WinLevels"] != executable["WinLevels"]:
                        raise AssertionError(f"WinLevels mismatch for {model}/{effort}/run{run}/{game}")
                    text_rhae = float(text["RHAEPercent"])
                    exec_rhae = float(executable["RHAEPercent"])
                    delta = exec_rhae - text_rhae
                    pairs.append(
                        {
                            "Game": game,
                            "Model": model,
                            "Effort": effort,
                            "Run": run,
                            "TextDirectory": text["Directory"],
                            "ExecutableDirectory": executable["Directory"],
                            "TextState": text["State"],
                            "ExecutableState": executable["State"],
                            "TextVerifiedSolved": solved(text),
                            "ExecutableVerifiedSolved": solved(executable),
                            "TextLevelsCompleted": int(text["LevelsCompleted"]),
                            "ExecutableLevelsCompleted": int(executable["LevelsCompleted"]),
                            "WinLevels": int(text["WinLevels"]),
                            "TextActions": int(text["GlobalStep"]),
                            "ExecutableActions": int(executable["GlobalStep"]),
                            "TextRHAEPercent": f"{text_rhae:.9f}",
                            "ExecutableRHAEPercent": f"{exec_rhae:.9f}",
                            "ExecutableMinusTextPP": f"{delta:.9f}",
                            "AbsoluteDifferencePP": f"{abs(delta):.9f}",
                            "OutcomeClass": outcome_class(text, executable),
                        }
                    )

    pairs.sort(
        key=lambda row: (
            -float(row["AbsoluteDifferencePP"]),
            row["Game"],
            row["Model"],
            row["Effort"],
        )
    )

    pairs_by_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        pairs_by_game[row["Game"]].append(row)

    summaries: list[dict[str, Any]] = []
    for game, rows in pairs_by_game.items():
        max_row = max(rows, key=lambda row: float(row["AbsoluteDifferencePP"]))
        deltas = [float(row["ExecutableMinusTextPP"]) for row in rows]
        classes = [row["OutcomeClass"] for row in rows]
        summaries.append(
            {
                "Game": game,
                "MatchedCells": len(rows),
                "MeanExecutableMinusTextPP": f"{fmean(deltas):.9f}",
                "MeanAbsoluteDifferencePP": f"{fmean(abs(value) for value in deltas):.9f}",
                "MaxAbsoluteDifferencePP": max_row["AbsoluteDifferencePP"],
                "MaxDifferenceModel": max_row["Model"],
                "MaxDifferenceEffort": max_row["Effort"],
                "MaxDifferenceSignedPP": max_row["ExecutableMinusTextPP"],
                "MaxDifferenceOutcomeClass": max_row["OutcomeClass"],
                "ExecutableBetterCells": sum(value > 0 for value in deltas),
                "TextBetterCells": sum(value < 0 for value in deltas),
                "TiedCells": sum(value == 0 for value in deltas),
                "ExecutableWinFlipCells": classes.count("executable_win_flip"),
                "TextWinFlipCells": classes.count("text_win_flip"),
                "BothSolvedCells": classes.count("both_solved"),
            }
        )

    summaries.sort(
        key=lambda row: (
            -float(row["MaxAbsoluteDifferencePP"]),
            -float(row["MeanAbsoluteDifferencePP"]),
            row["Game"],
        )
    )

    expected_pairs = len(models) * len(efforts) * len(runs) * len(games)
    if len(pairs) != expected_pairs:
        raise AssertionError(f"expected {expected_pairs} matched cells, found {len(pairs)}")
    if any(len(rows) != len(models) * len(efforts) * len(runs) for rows in pairs_by_game.values()):
        raise AssertionError("not every game has complete model/effort/run coverage")

    write_rows(args.pairs_output, pairs, PAIR_FIELDS)
    write_rows(args.summary_output, summaries, SUMMARY_FIELDS)
    print(f"source_rows={len(source_rows)}")
    print(f"matched_cells={len(pairs)}")
    print(f"games={len(summaries)}")
    print(f"models={','.join(models)} efforts={','.join(efforts)} runs={','.join(runs)}")
    print(f"pairs_output={args.pairs_output}")
    print(f"summary_output={args.summary_output}")


if __name__ == "__main__":
    main()
