#!/usr/bin/env python3
"""Offline checks of the pinned public patch. No model or game execution.

Rebuilds the patch in a temporary directory, extracts three pure definitions
with AST, and exercises them on synthetic grids with independently known roles.
Prints JSON to stdout. Requires Python 3 and the system patch command.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / "SOURCES.json").read_text())
    for item in manifest["files"]:
        assert hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() == item["sha256"], item["path"]
    base = ROOT / "sources/harnesses/baseline-v12/src/ARC3-Inference"
    patch = ROOT / "sources/harnesses/ffa7g/patch/ffa7g-full-stack.patch"
    with tempfile.TemporaryDirectory(prefix="duck-source-check-") as tmp:
        rebuilt = Path(tmp) / "src"
        shutil.copytree(base, rebuilt)
        applied = subprocess.run(
            ["patch", "--batch", "--fuzz=0", "-p1", "-i", str(patch)],
            cwd=rebuilt, text=True, capture_output=True, check=True,
        )
        solver = rebuilt / "inference/framework/solver.py"
        tree = ast.parse(solver.read_text())
        names = {"_HousekeepingBand", "_changed_axes", "_verify_hud_model"}
        definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
        assert {node.name for node in definitions} == names
        module = ast.Module(body=definitions, type_ignores=[])
        ns = {}
        exec(compile(module, str(solver), "exec"), ns)
        Band, axes, verify = (ns[name] for name in ("_HousekeepingBand", "_changed_axes", "_verify_hud_model"))

        def avatar(col, row=10):
            grid = [[0] * 64 for _ in range(64)]
            grid[row][col] = 1
            return grid

        def classify(band, before, after):
            rows, cols = band.band()
            changed_rows, changed_cols, outside = axes(before, after, rows, cols)
            # Same order and predicate as the reconstructed _execute_action path.
            gameplay_changed = outside if rows or cols else before != after
            band.observe(changed_rows, changed_cols)
            return {"rows": sorted(rows), "cols": sorted(cols), "no_impact": not gameplay_changed}

        band = Band()
        warmup = [classify(band, avatar(c), avatar(c + 1)) for c in range(8)]
        assert all(not x["no_impact"] for x in warmup)
        movement = classify(band, avatar(8), avatar(9))
        assert movement == {"rows": [10], "cols": [], "no_impact": True}
        # Control: movement off the learned row remains visible.
        off_row = classify(band, avatar(9), avatar(9, row=11))
        assert off_row["no_impact"] is False
        band.reset()
        after_reset = classify(band, avatar(8), avatar(9))
        assert after_reset["no_impact"] is False

        # Positive control: a synthetic HUD has the same frequency signature.
        hud = Band()
        def counter(n):
            grid = [[0] * 64 for _ in range(64)]
            grid[63][:n] = [1] * n
            return grid
        for n in range(8):
            classify(hud, counter(n), counter(n + 1))
        hud_only = classify(hud, counter(8), counter(9))
        assert hud_only == {"rows": [63], "cols": [], "no_impact": True}

        # Matching a prediction does not establish that predicted cells are HUD.
        before, actual = ["WwWW"], ["WWwW"]
        accepted, mask = verify(before, actual, actual)
        _, _, remains = axes(before, actual, set(), set(), mask)
        assert accepted and mask == {(0, 1), (0, 2)} and not remains

        old_actions = runpy.run_path(str(base / "inference/agent/action_names.py"))
        new_actions = runpy.run_path(str(rebuilt / "inference/agent/action_names.py"))
        old_a7 = old_actions["to_engine_action"]("ACTION7")
        new_a7 = new_actions["to_engine_action"]("ACTION7")
        assert old_a7 is None and new_a7 == "ACTION7"
        # Current patch contains the graph despite a later manifest saying it was absent.
        graph_present = any(isinstance(node, ast.ClassDef) and node.name == "_StateGraph" for node in tree.body)
        assert graph_present

        facts = {
            "scope": "Offline source reconstruction and synthetic counterexamples; no scored-run reproduction",
            "commit": manifest["repository_commit"],
            "verified_source_files": len(manifest["files"]),
            "patch_exit": applied.returncode,
            "patch_fuzz": 0,
            "reconstructed_solver_sha256": hashlib.sha256(solver.read_bytes()).hexdigest(),
            "movement_false_no_impact": {"transition": "avatar moves from row 10 column 8 to row 10 column 9 after eight horizontal moves", **movement},
            "movement_off_mask_control": off_row,
            "level_reset_control": after_reset,
            "hud_positive_control": hud_only,
            "accurate_movement_predictor_accepted_as_hud": {"accepted": accepted, "masked_cells": sorted(mask), "change_outside_mask": remains},
            "action7_roundtrip": {"baseline": old_a7, "patched": new_a7},
            "state_graph_present_in_committed_patch": graph_present,
            "reported_noimpact_pass_scores": {"off": [0.946, 1.145], "on": [2.061, 1.187]},
            "reported_pass_mean_gain_percent": (sum([2.061, 1.187]) / sum([0.946, 1.145]) - 1) * 100,
        }
        # Recheck the related historical local failure without rewriting its evidence.
        local = json.loads((ROOT / "LOCAL-REPLAY.json").read_text())
        repo = ROOT.parents[1]
        local_paths = [repo / item["path"] for item in local["source_files"]]
        for item, path in zip(local["source_files"], local_paths):
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        local_tree = ast.parse(local_paths[0].read_text())
        record_class = next(node for node in local_tree.body if isinstance(node, ast.ClassDef) and node.name == "ObservationRecord")
        record_class.decorator_list = []
        local_ns = {}
        exec(compile(ast.Module(body=[record_class], type_ignores=[]), str(local_paths[0]), "exec"), local_ns)
        record = local_ns["ObservationRecord"]()
        for line in local_paths[1].read_text().splitlines():
            transition = json.loads(line)
            before, after = transition["before"]["grid"], transition["after"]["grid"]
            record.observe(before, after)
            if transition["step"] == 5:
                changed = sum(x != y for a, b in zip(before, after) for x, y in zip(a, b))
                assert changed == 52 and record.classify(before, after) == "hud_only"
                facts["local_historical_replay"] = {"step": 5, "changed_pixels": changed, "label": record.classify(before, after)}
                break
        assert "local_historical_replay" in facts
        print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()
