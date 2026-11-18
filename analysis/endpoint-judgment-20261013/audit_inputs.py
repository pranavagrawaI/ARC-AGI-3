"""Reproduce baseline denominators and verify all starts without any model calls."""
import ast
import json
import os
from pathlib import Path
import sys

import screen


def audit():
    root = screen.HERE / "evidence"
    root.mkdir(exist_ok=True)
    cfg = screen.config(screen.parser().parse_args(["--dry"]))
    os.environ.update(screen.child_env(root / "input-audit", cfg))
    sys.path[:0] = os.environ["PYTHONPATH"].split(os.pathsep)
    from starts import play, fhash
    from arcengine import GameState
    verified = []
    for key, s in screen.starts().items():
        _, fr = play(s["gid"], s["prefix"])
        from arcengine import ActionInput, GameAction
        g, before = play(s["gid"], s["prefix"])
        after = g.perform_action(ActionInput(id=GameAction.RESET))
        verified.append({"key": key, "level": fr.levels_completed, "frame_hash": fhash(fr),
                         "valid": fr.levels_completed == s["level"] and fr.state != GameState.GAME_OVER
                         and fhash(fr) in s["entry_hashes"].values(), "klass": s["klass"],
                         "reset_keeps_level_and_entry": after.levels_completed == s["level"] and fhash(after) == fhash(before)})
    screen.atomic(root / "starts-verification.json", {"verified": sum(r["valid"] for r in verified), "selected": len(verified), "rows": verified})

    bm_path = screen.ORACLE / "sessions/a0/output/benchmark.json"
    bm = screen.load(bm_path)
    readout = screen.load(screen.ORACLE / "readout.json")
    rows = []
    parse = lambda v: v if isinstance(v, list) else ast.literal_eval(v)
    for r in bm["game_runs"]:
        key = r["game_id"].split("-A0-")[0]
        hist = parse(r["history"])
        final = int(r.get("final_generated_tokens") or 0)
        tokens = sum(int(h.get("generated_tokens") or 0) for h in hist) + final
        corrected = readout["rows"][r["game_id"]]
        log_path = bm_path.parent / f"{r['game_id']}_p0_requests.jsonl"
        raw_tokens = 0
        response_count = 0
        unknown_usage = 0
        with log_path.open() as stream:
            for line in stream:
                event = json.loads(line)
                if event.get("event") == "response":
                    response_count += 1
                    n = (event.get("usage") or {}).get("completion_tokens")
                    if isinstance(n, int):
                        raw_tokens += n
                    else:
                        unknown_usage += 1
        rows.append({"attempt": r["game_id"], "key": key, "solved": corrected["solved"],
                     "tokens": tokens, "actions": len(hist), "seconds": r["final_wallclock_seconds"],
                     "final_generated_tokens": final, "klass": screen.starts()[key]["klass"], "won": r["state"] == "won",
                     "raw_response_tokens": raw_tokens, "response_count": response_count, "unknown_usage_responses": unknown_usage,
                     "raw_requests_path": str(log_path)})
    def group(data, token_field="raw_response_tokens"):
        tokens, solved = sum(r[token_field] for r in data), sum(r["solved"] for r in data)
        return {"attempts": len(data), "unique_starts": len({r["key"] for r in data}), "solves": solved,
                "solve_rate": solved / len(data), "generated_tokens": tokens, "mean_generated_tokens": tokens / len(data),
                "actions": sum(r["actions"] for r in data), "solves_per_100k_generated_tokens": solved * 100000 / tokens,
                "token_source": token_field, "unknown_usage_responses": sum(r["unknown_usage_responses"] for r in data)}
    selected = {t["key"] for t in readout["table"] if t["p1"] is not None}
    baseline = {"all_40": group(rows), "hint_comparison_26": group([r for r in rows if r["key"] in selected]),
                "never": group([r for r in rows if r["klass"] == "never"]), "rows": rows,
                "benchmark_accounting_all_40": group(rows, "tokens"),
                "benchmark_accounting_hint_comparison_26": group([r for r in rows if r["key"] in selected], "tokens"),
                "evidence": [str(bm_path), str(screen.ORACLE / "readout.json"), str(screen.ORACLE / "sessions/a0/identity.json")],
                "cap_seconds": 1200, "budget_type": "time cap, not an 80K generated-token cap",
                "model_identity": "flashnext alias; serve.log reports intel-qwen3.8-flash-next-w4a16-autoround target, quant=auto-round bits=4, MTP drafter quant=compressed-tensors; not an NVFP4 target receipt",
                "serving_identity_evidence": str(screen.HERE / "evidence/a0-serving-identity.json")}
    screen.atomic(root / "baseline.json", baseline)
    print(json.dumps({k: v for k, v in baseline.items() if k != "rows"}, indent=2))
    if not all(r["valid"] and r["reset_keeps_level_and_entry"] for r in verified):
        raise SystemExit("start verification failed")


if __name__ == "__main__":
    audit()
