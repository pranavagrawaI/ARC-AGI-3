"""CPU counterexample to occupancy-gated fixed-gap speed screening.

Uses the existing simulator without changing its files or refitting alpha. All
corpus gaps are built ONCE using the original control rates, then frozen.
Multipliers are hypothetical decode rates, not measured candidate performance.
"""
import copy
import hashlib
import json
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OLD=ROOT/"analysis/mtp-depth-20261006"
sys.path.insert(0,str(OLD))
import sim


def main():
    path=OLD/"kaggle/dataset/workload.json"
    games=sim.build(json.loads(path.read_text()))
    original=sim.per_stream_rate
    original_from=sim.MEASURE_FROM_S
    sim.MEASURE_FROM_S=600
    results=[]
    try:
        for alpha in (.523,0.):
            rows=[]
            for multiplier in (1.,1.1,1.2,1.3,1.5):
                sim.per_stream_rate=lambda r,m=multiplier:original(r)*m
                result=sim.simulate(copy.deepcopy(games),alpha,horizon=1500)
                result["decode_multiplier"]=multiplier
                result["absolute_band_pass"]=8.57<=result["mean_R"]<=9.37 and .62<=result["p_R_ge_9"]<=.82
                rows.append(result)
            for row in rows:
                row["paired_mean_delta"]=abs(row["mean_R"]-rows[0]["mean_R"])
                row["paired_p9_delta"]=abs(row["p_R_ge_9"]-rows[0]["p_R_ge_9"])
                row["paired_band_pass"]=row["paired_mean_delta"]<=.25 and row["paired_p9_delta"]<=.05
            results.append(dict(alpha=alpha,rows=rows))
    finally:
        sim.per_stream_rate=original;sim.MEASURE_FROM_S=original_from
    report=dict(scope="Hypothetical CPU simulator sensitivity; NOT measured engine rate or native residence qualification",
                source_sha256=hashlib.sha256((OLD/"sim.py").read_bytes()).hexdigest(),
                corpus_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                gaps_built_once_under_control=True,window_s=[600,1500],results=results,
                caveats=["Simulator decode phase differs from actual native decode residence and legacy gauges",
                         "Rounded simulator output and set iteration order can perturb exact rows",
                         "alpha=0 is a proposed distinct saturated workload; old production bands do not apply",
                         "Per-stream normalization cannot repair outcome-dependent admission"])
    (HERE/"occupancy-sensitivity.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))


if __name__=="__main__":main()
