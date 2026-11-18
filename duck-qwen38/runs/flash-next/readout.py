"""Compare an arm's Kaggle output dir with KF (out-kf2): levels per game plus mechanism checks.

  python3 readout.py out-r1 [out-s1 ...]
"""
import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ACTION7_GAMES = ("ar25", "bp35", "lf52", "sb26", "sk48", "su15")


def summarize(out: Path) -> dict:
    bench = json.loads((out / "benchmark.json").read_text())
    runs = {r["game_id"][:4]: r for r in bench["game_runs"]}
    score = json.loads((out / "score.json").read_text()) if (out / "score.json").is_file() else {}
    per_game_score = {k[:4]: v["score"] for k, v in score.get("games", {}).items()}
    action7, analyses, yielded = {}, 0, 0
    for path in (out / "artifacts").glob("*_events.jsonl"):
        game = path.name[:4]
        for line in path.open():
            event = json.loads(line)
            if event["type"] == "action" and event.get("action_name") == "ACTION7":
                action7[game] = action7.get(game, 0) + 1
            elif event["type"] == "analysis":
                analyses += 1
                yielded += "yield" in json.dumps(event.get("status", "")).lower()
    log = (out / "vllm-openai-server.log").read_text(errors="replace") if (out / "vllm-openai-server.log").is_file() else ""
    kv = re.search(r"GPU KV cache size: ([\d,]+) tokens", log)
    samples = [(int(r), int(w), float(g)) for g, r, w in re.findall(
        r"Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs", log)]
    busy = [s for s in samples if s[0] + s[1] >= 20]
    return {
        "levels": sum(r["levels_completed"] for r in runs.values()),
        "games_with_level": sum(r["levels_completed"] > 0 for r in runs.values()),
        "actions": sum(len(r["history"]) for r in runs.values()),
        "mean_score": statistics.fmean(per_game_score.values()) if per_game_score else None,
        "per_game": {g: runs[g]["levels_completed"] for g in sorted(runs)},
        "action7_executed": action7,
        "kv_tokens": int(kv[1].replace(",", "")) if kv else None,
        "busy_median_running": statistics.median(s[0] for s in busy) if busy else None,
        "busy_median_waiting": statistics.median(s[1] for s in busy) if busy else None,
        "busy_median_gen_tok_s": statistics.median(s[2] for s in busy) if busy else None,
        "states": sorted({r["state"] for r in runs.values()}),
    }


def main() -> None:
    base = summarize(HERE / "out-kf2")
    arms = {"KF": base, **{Path(a).name: summarize(HERE / a) for a in sys.argv[1:]}}
    keys = ["levels", "games_with_level", "actions", "mean_score", "kv_tokens",
            "busy_median_running", "busy_median_waiting", "busy_median_gen_tok_s", "states"]
    print("| metric | " + " | ".join(arms) + " |\n|---|" + "---|" * len(arms))
    for k in keys:
        print(f"| {k} | " + " | ".join(str(round(v[k], 2) if isinstance(v[k], float) else v[k]) for v in arms.values()) + " |")
    print("\nper-game levels (only games that differ from KF):")
    for g in base["per_game"]:
        row = {n: a["per_game"].get(g) for n, a in arms.items()}
        if len(set(row.values())) > 1:
            print(f"  {g}: " + "  ".join(f"{n}={v}" for n, v in row.items()))
    for n, a in arms.items():
        print(f"{n} ACTION7 executed: {a['action7_executed']} (advertised in {ACTION7_GAMES})")


if __name__ == "__main__":
    main()
