"""Independent source checks, parent integrity, and clock-origin receipts."""
import datetime
import hashlib
import importlib.util
import json
import math
import re
import sys

import heldout as h


def main():
    target = h.HERE / "VERIFICATION.json"
    assert not target.exists(), target
    manifest = json.loads((h.HERE / "MANIFEST.json").read_text())
    for name, digest in manifest["input_sha256"].items():
        assert hashlib.sha256((h.ROOT / name).read_bytes()).hexdigest() == digest, name
    assert hashlib.sha256((h.HERE / "heldout.py").read_bytes()).hexdigest() == manifest["code_sha256"]
    source = h.ROOT / "analysis/counterexample-continuations-v3-20261013/output/continuations/sources/control/inference/agent/priority_scheduler.py"
    spec = importlib.util.spec_from_file_location("reference_priority", source)
    ref = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ref
    spec.loader.exec_module(ref)
    checks = 0
    for n in range(6, 11):
        for k in (0, 1, 3, 6, 10):
            for actions in (0, 10, 115, 1000):
                for tokens in (0, 20_000, 62_000, 124_000, 1_000_000):
                    for tail in (None, 0.0, 0.1, 0.5, 1.0):
                        expected = ref.priority_value(ref.PrioritySnapshot(k, actions, tokens, total_levels=n),
                                                      normalize_score=True, tail_lookup="remaining", tail_fraction=tail)
                        actual = h.sim.priority_value(k, actions, tokens, n, tail)
                        assert actual == expected, (n, k, actions, tokens, tail, actual, expected)
                        checks += 1
    data = json.loads((h.PARENT / "attempts.json").read_text())
    score_checks = 0
    clock = []
    event_checks = 0
    artifact_sha = {}
    for run, meta in data["runs"].items():
        directory = h.ROOT / meta["dir"]
        p = directory / "benchmark.json"
        artifact_sha[str(p.relative_to(h.ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
        raw = json.loads(p.read_text())
        for game in raw["game_runs"]:
            predicted = h.sim.game_score(game["number_of_levels"], game["base_actions_per_level"],
                                         game["levels_completed"], game["actions_per_level"])
            assert math.isclose(predicted, game["final_score"], abs_tol=1e-7), (run, game["game_id"])
            score_checks += 1
            if meta["slots"] != 14:
                continue
            gid = game["game_id"][:4]
            transcript = next((directory / "transcripts").glob(gid + "*.txt"))
            header = re.search(r"^--- analysis_step=\d+ \| action=\d+ \| (\d\d:\d\d:\d\d)", transcript.read_text(), re.M)
            started = datetime.datetime.fromisoformat(game["started_at"])
            first = datetime.datetime.combine(started.date(), datetime.datetime.strptime(header[1], "%H:%M:%S").time())
            offset = (first - started).total_seconds()
            if offset < -1:
                offset += 86400
            clock.append({"run": run, "game": gid, "game_start": game["started_at"],
                          "first_transcript_turn": first.isoformat(), "first_wait_s": offset})
            completed = 0
            charged_actions = []
            total = 0
            for a in game["actions_per_level"][:game["levels_completed"]]:
                total += a
                charged_actions.append(total)
            events = next((directory / "artifacts").glob(gid + "*_events.jsonl"))
            for line in events.open():
                e = json.loads(line)
                if e.get("type") == "action" and int(e.get("score") or 0) > completed:
                    new = int(e["score"])
                    assert new == completed + 1
                    assert e["action_num"] == charged_actions[completed], (run, gid)
                    completed = new
                    event_checks += 1
            assert completed == game["levels_completed"]
    result = json.loads((h.HERE / "RESULT.json").read_text())
    pair_checks = 0
    max_token_delta = 0.0
    for comparison in result["comparisons"]:
        path = h.HERE / "runs" / f"{comparison['cohort']}-{comparison['scenario']}-fold-{comparison['fold']}.json"
        rows = json.loads(path.read_text())["rows"]
        for row in rows:
            a, b = row["shipped"], row["hazard"]
            delta = abs(a["tokens"] - b["tokens"])
            max_token_delta = max(max_token_delta, delta)
            for arm in (a, b):
                assert math.isclose(arm["tokens"], h.SLOTS * h.RATE * h.HORIZON, abs_tol=1e-4)
                assert math.isclose(sum(arm["provenance_score"].values()), arm["score"], abs_tol=1e-7)
                if comparison["scenario"] in ("tail_zero", "empirical_only"):
                    assert arm["extrapolated_levels"] == arm["depends_on_extrapolation_score"] == 0
            assert delta < 1e-4
            pair_checks += 1
    out = {"protected_parent_hashes_unchanged": True, "priority_source": str(source.relative_to(h.ROOT)),
           "priority_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "priority_cases_matched": checks,
           "saved_game_scores_matched": score_checks, "level_completion_event_boundaries_matched": event_checks,
           "paired_runs_checked": pair_checks, "max_paired_token_difference": max_token_delta,
           "artifact_sha256": artifact_sha, "clock_receipts": clock}
    target.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("artifact_sha256", "clock_receipts")}), flush=True)


if __name__ == "__main__":
    main()
