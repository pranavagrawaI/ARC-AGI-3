"""Recompute the source identities and downloaded demo facts without running notebooks."""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import pathlib
import re
import statistics
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    result = {"audited_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    result["notebooks"] = {}
    for who in ("dfranzen", "lordhansolo", "sirikilohit"):
        original = next((ROOT.parent / who).glob("*.ipynb"))
        fresh = next((ROOT / f"live-{who}").glob("*.ipynb"))
        meta = json.loads((fresh.parent / "kernel-metadata.json").read_text())
        result["notebooks"][who] = {
            "url": "https://www.kaggle.com/code/" + meta["id"],
            "sha256": digest(fresh),
            "matches_preexisting_download": digest(original) == digest(fresh),
            "input_datasets": meta["dataset_sources"],
            "input_models": meta["model_sources"],
            "machine_shape": meta["machine_shape"],
            "note": "Latest source pulled live; Kaggle pull metadata does not pin a kernel version.",
        }

    result["dfranzen_git_head"] = subprocess.check_output(
        ["git", "-C", str(ROOT / "dfranzen-repo"), "rev-parse", "HEAD"], text=True
    ).strip()
    overlay = ROOT / "lord-runtime/arc3_vllm_main_e975732_arc3_overlay.tar.blob"
    expected = "e3a6fe0d9f010bc1fb66e53e7dc6fde43272824d5200da5a49b5c4456065e0cb"
    result["lord_runtime_overlay"] = {"sha256": digest(overlay), "matches_launcher_pin": digest(overlay) == expected}
    assert result["lord_runtime_overlay"]["matches_launcher_pin"]

    result["downloaded_demos"] = {}
    for who in ("dfranzen", "lord", "siri"):
        directory = ROOT / f"{who}-output"
        data = json.loads((directory / "benchmark.json").read_text())
        games = data["game_runs"]
        elapsed = (dt.datetime.fromisoformat(data["end_time"]) - dt.datetime.fromisoformat(data["start_time"])).total_seconds()
        generated = sum(
            sum(h.get("generated_tokens", 0) for h in g["history"]) + g.get("final_generated_tokens", 0)
            for g in games
        )
        result["downloaded_demos"][who] = {
            "games": len(games), "games_with_actions": sum(bool(g["history"]) for g in games),
            "seconds": elapsed, "levels": sum(g["levels_completed"] for g in games),
            "actions": sum(len(g["history"]) for g in games),
            "mean_score": statistics.mean(g["final_score"] for g in games),
            "generated_tokens_accounted_in_benchmark": generated,
            "accounted_generation_per_wall_second": generated / elapsed,
            "states": {s: sum(g["state"] == s for g in games) for s in sorted({g["state"] for g in games})},
            "warning": "Unequal short demo budgets; not leaderboard runs, causal comparisons, or matched full public-25 passes.",
        }

    log = (ROOT / "lord-output/vllm-openai-server.log").read_text()
    rows = [tuple(map(float, m)) for m in re.findall(
        r"Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%, Prefix cache hit rate: ([\d.]+)%", log
    )]
    result["lord_log"] = {
        "kv_token_capacity": int(re.search(r"GPU KV cache size: ([\d,]+) tokens", log)[1].replace(",", "")),
        "periodic_samples": len(rows),
        "median_generation_tokens_per_second": statistics.median(r[0] for r in rows),
        "median_running": statistics.median(r[1] for r in rows),
        "median_waiting": statistics.median(r[2] for r in rows),
        "last_logged_prefix_hit_percent": rows[-1][4],
        "caveat": "Logger's reported hit statistic; not independently reconstructed token-weighted cache reuse.",
    }

    # Exercise the small imported retention implementation, using only authored toy code.
    retained = load_module("donor_retention", ROOT / "dfranzen-repo/ARC3-Inference/inference/utils/retained_functions.py")
    sources = ["def count_cells(board):\n    return len(board)",
               "def stale():\n    return old_board",
               "def snapshot(board=current_frame):\n    return board"]
    kept, rejected = retained.prepare_functions(sources, {"len"})
    namespace = {}
    retained.restore_functions(list(kept.values()), namespace, {"len"})
    assert list(kept) == ["count_cells"]
    assert namespace["count_cells"]([1, 2]) == 2
    assert namespace["count_cells"]([1, 2, 3]) == 3
    result["retention_probe"] = {
        "retained": list(kept), "rejected": rejected,
        "fresh_observations_produce": [2, 3],
        "scope": "CPU unit probe only; no sandbox integration or gameplay qualification.",
    }
    result["files"] = {
        str(p.relative_to(ROOT)): {"bytes": p.stat().st_size, "sha256": digest(p)}
        for p in sorted(ROOT.rglob("*"))
        if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts
        and p.name not in {"evidence.json", "REPORT.md"}
    }
    (ROOT / "evidence.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
