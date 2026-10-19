"""Read-only CPU attribution checks. Run: python3 -B analysis/d7-attribution-20261017/analyze.py."""
import hashlib
import ast
import json
import re
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(path):
    return json.loads((ROOT / path).read_text())


def code_cells(path):
    return ["".join(c.get("source", [])) for c in read(path)["cells"] if c["cell_type"] == "code"]


def main():
    ids = [("D0", 56747536), ("D1", 57039090), ("D2", 57056746),
           ("D3", 57099732), ("D4", 57155601), ("D5", 57175066), ("D7", 57265770)]
    rows = []
    for label, ref in ids:
        record = json.loads((HERE / f"submission-{ref}.json").read_text())
        d = record["detail"]
        assert d["status"] == "COMPLETE", (ref, d["status"])
        rows.append(dict(draw=label, submission_id=ref, score=float(d["publicScore"]),
                         control=label in {"D0", "D2", "D4", "D7"}, date=d["date"]))
    control = [r["score"] for r in rows if r["control"]]
    by_draw = {r["draw"]: r["score"] for r in rows}
    pooled = [r["score"] for r in read("analysis/notebook-mining-20261016/family-draws.json")
              if r["cls"] == "plain"]
    pairs = []
    for candidate, before, after in [("D1", "D0", "D2"), ("D3", "D2", "D4"), ("D5", "D4", "D7")]:
        neighbor_mean = st.mean([by_draw[before], by_draw[after]])
        pairs.append(dict(candidate=candidate, controls=[before, after], control_mean=neighbor_mean,
                          candidate_minus_neighbor_mean=by_draw[candidate] - neighbor_mean))

    raw_local = "analysis/m2-winners-20261001/runs/raw/dfranzen-m2-raw.ipynb"
    raw_remote = str(next((HERE / "remote-current").glob("*.ipynb")).relative_to(ROOT))
    base = "solvers/m2_b1/experiment/pair-1"
    notebooks = [f"{base}/b1-r{i}/kernel/m2-b1-pair1-r{i}.ipynb" for i in [1, 2]]
    benchmarks = [f"{base}/b1-r{i}/output/benchmark.json" for i in [1, 2]]
    logs = [f"{base}/b1-r{i}/output/m2-b1-pair1-r{i}.log" for i in [1, 2]]
    installed = [ast.literal_eval(re.search(r"M2 B1 installed (.*?)\\n", (ROOT / p).read_text())[1])
                 for p in logs]
    runs = [{r["game_id"]: r for r in read(p)["game_runs"]} for p in benchmarks]
    assert len(runs[0]) == len(runs[1]) == 25 and runs[0].keys() == runs[1].keys()
    per_game = []
    summaries = []
    for run in runs:
        summaries.append(dict(levels=sum(r["levels_completed"] for r in run.values()),
                              mean_score=st.mean(r["final_score"] for r in run.values()),
                              generated_tokens=sum(sum(h.get("generated_tokens", 0) for h in r["history"])
                                                   + r.get("final_generated_tokens", 0) for r in run.values())))
    for game, a in runs[0].items():
        b = runs[1][game]
        per_game.append(dict(game=game, levels_r1=a["levels_completed"], levels_r2=b["levels_completed"],
                             score_r1=a["final_score"], score_r2=b["final_score"],
                             contribution_to_mean_gap=(b["final_score"] - a["final_score"]) / 25))
    per_game.sort(key=lambda r: r["contribution_to_mean_gap"], reverse=True)
    sources = [raw_local, raw_remote, *notebooks, *benchmarks, *logs,
               "analysis/notebook-mining-20261016/family-draws.json",
               "analysis/cn04-b1-inspection-20261012/findings.json"]
    result = dict(
        observations=rows,
        own_control=dict(n=len(control), mean=st.mean(control), sample_sd=st.stdev(control),
                         minimum=min(control), maximum=max(control),
                         b1_minus_mean=by_draw["D5"] - st.mean(control),
                         d7_minus_b1=by_draw["D7"] - by_draw["D5"],
                         d7_minus_d2=by_draw["D7"] - by_draw["D2"]),
        pooled_context=dict(n=len(pooled), mean=st.mean(pooled), sample_sd=st.stdev(pooled),
                            d7_standardized_difference=(by_draw["D7"] - st.mean(pooled)) / st.stdev(pooled),
                            at_least_d7=sum(x >= by_draw["D7"] for x in pooled),
                            caveat="External self-selected notebook scores with mixed version bindings; descriptive context, not iid inference."),
        neighboring_comparisons=pairs,
        raw_remote_code_matches_local=code_cells(raw_local) == code_cells(raw_remote),
        b1_replicate_code_equal=code_cells(notebooks[0]) == code_cells(notebooks[1]),
        b1_installed_overlay_hashes_equal=installed[0] == installed[1],
        b1_replicates=summaries,
        b1_generated_token_ratio=summaries[1]["generated_tokens"] / summaries[0]["generated_tokens"],
        b1_score_gap=summaries[1]["mean_score"] - summaries[0]["mean_score"],
        b1_per_game=per_game,
        source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        limits=["Hidden per-game logs unavailable in these probes: episode lists empty; downloads HTTP 403.",
                "No estimate of the probability that B1 is worse; it has one hidden draw.",
                "Public B1 replicate variation demonstrates a mechanism and scale, not D7-specific causation.",
                "Identical source and pinned inputs do not establish identical effective hardware speed or scheduling."])
    (HERE / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"b1_per_game", "source_sha256"}}, indent=2))


if __name__ == "__main__":
    main()
