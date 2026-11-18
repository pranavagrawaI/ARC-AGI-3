"""Pooled hidden-set draws for dfranzen-harness notebooks, classified by rerun-path delta.
Scores scraped 2026-10-16 from notebook pages (lists/scores-*.jsonl); classification from lists/family.jsonl + manual diff review.
usage: python3 -B draw_ledger.py > family-draws.json (also prints summary to stderr)"""
import json, statistics as st, sys, math
D = [  # (ref, score, class, binding) binding: 'current' = page Public Score of the version whose code we pulled
 ("dfranzen/arc-agi-3-milestone-2-solution", 33.68, "plain", "current"),
 ("jingshuifu/arc3-m2-solution", 33.22, "plain", "current"),
 ("greenalgeria/arc-agi-3-milestone-final", 27.13, "plain", "current"),
 ("greenalgeria/arc-agi-3-milestone-final", 32.38, "plain", "best-V1, code assumed identical (0 added lines in V2)"),
 ("zhangyues0000/arc-agi-3-m2-franzen-fork", 31.76, "plain", "current"),
 ("sigeward/arc-agi-3-milestone-2-solution", 31.27, "plain", "current; 120 s teardown reserve in rerun"),
 ("spark328/arc-prize-2026-arc-agi-3-franzen-v1", 30.75, "plain", "current"),
 ("lwq255/arc3-dfranzen-m2-guarded", 30.63, "plain", "current; placeholder guard"),
 ("manwithacat/digital-mouse-qwen-genome-baseline", 31.07, "plain", "current; own python runtime"),
 ("hknight888/hknight3-0", 28.52, "plain", "current"),
 ("juliancamilovilla/arc-agi3-franzen-m2", 27.63, "plain", "current"),
 ("jamesmills1975/cage-x1r3-r11-submit", 27.45, "plain", "V1 of 2; patch hash pinned to dfranzen"),
 ("lizw1233/arc-agi-3-milestone-2-solution", 27.28, "plain", "current"),
 ("sad12454/arc-agi-3-franzen-m2-reproduction", 26.70, "plain", "current"),
 ("kunaldesale2408/duck-harness-fast-eval", 26.44, "plain", "current"),
 ("felixwang1m/arc-agi-3-m2-open-source-reproduction-franzen", 26.00, "plain", "current"),
 ("viekilee/arc-agi-3-m2-verified-build", 25.83, "plain", "current; 120 s kill reserve"),
 ("lelexihongshi/arc-agi-3-milestone-2-solution", 24.60, "plain", "current"),
 ("svegio/arc-agi-3-milestone-2-solution", 24.28, "plain", "current"),
 ("anthonyxlyu/arc-agi-3-milestone-2-solution", 24.21, "plain", "current"),
 ("argszeroreg/arc-agi-3-dfranzen-clean", 23.71, "plain", "current"),
 ("dfranzen (LB, 10-01)", 27.89, "plain", "external record"),
 ("affectify unchanged control", 27.62, "plain", "external record (survey 10-11)"),
 ("ours control D0", 23.87, "plain", "hidden-draw-ledger"),
 ("ours control", 29.44, "plain", "hidden-draw-ledger"),
 ("ours control", 24.97, "plain", "hidden-draw-ledger"),
 ("amatlas/dfranzen-fork-d-prime", 32.12, "dprime", "current"),
 ("lwq255/arc3-dprime-guarded", 29.15, "dprime", "current"),
 ("spark328/arc-prize-2026-arc-agi-3-affectify-v1", 22.17, "dprime", "current"),
 ("shiiin9/affectify-arc-31-54-in-a-single-sub", 31.54, "dprime", "best V1 of 2"),
 ("hitarthjain0/arc-agi-3-milestone-2-solution", 25.10, "dprime", "best V1 of 4 (weak binding)"),
 ("jingshuifu/arc3-m2-solution-v3", 32.60, "pace_only", "current"),
 ("jingshuifu/arc3-m2-solution-v2", 28.00, "tail_lookup_only", "current"),
 ("hknight888/hknight3-opt", 22.17, "temperature_0.4", "current"),
 ("juliancamilovilla/arc-agi3-e1-med", 20.98, "reasoning_effort_medium", "current"),
 ("gedouluhui/arc3-kv97r11-sub", 5.49, "memfrac0.97_11streams", "current"),
 ("habanwer/arcaid3-lexicon-view-dfranzen-m2", 31.90, "lexicon_view", "current"),
 ("desihnurseha/kinetic-learning-by-playing-v8", 29.68, "mechanics_catalog", "current"),
 ("ashwanidhukra/arc-agi3-win-v2", 29.13, "no_progress_recovery+programmatic_memory", "current"),
 ("juliancamilovilla/arc-agi3-franzen-h1", 24.17, "h1_tool_agent_patch", "current"),
 ("kozykappa/arc-agi-3-flashnext-duck-audited-submission", 25.47, "audit+priority_scales", "current"),
 ("themightiestman/arc3-yukailiu-dprime", 28.24, "dprime+pace+reset+temp0.70", "current"),
 ("yukailiu/arc-agi-3-champion-solution", 23.32, "dprime+dhia0.1.21+reset", "current"),
 ("mayakaripel/arc-agi-3-champion-solution", 20.53, "dprime+dhia0.1.21+reset", "current"),
 ("ouyangruize/arc-agi-3-milestone-2-solution", 21.17, "dprime+animation_noop_fix+static_hint", "current"),
]
out = [dict(ref=r, score=s, cls=c, binding=b) for r, s, c, b in D]
print(json.dumps(out, indent=1))
def summ(name, xs):
    m = st.mean(xs); sd = st.stdev(xs) if len(xs) > 1 else float('nan')
    print(f"{name:42s} n={len(xs):2d} mean={m:5.2f} sd={sd:4.2f} se={sd/math.sqrt(len(xs)):4.2f} min={min(xs):5.2f} max={max(xs):5.2f}", file=sys.stderr)
    return m, sd
plain = [s for _, s, c, _ in D if c == "plain"]
pm, psd = summ("plain dfranzen (all)", plain)
summ("plain, public notebooks only", [s for r, s, c, _ in D if c == "plain" and "/" in r])
dp = [s for _, s, c, _ in D if c == "dprime"]; summ("D-prime only", dp)
summ("D-prime only, current-bound", [s for _, s, c, b in D if c == "dprime" and b == "current"])
summ("any D-prime (incl. bundles)", [s for _, s, c, _ in D if c.startswith("dprime")])
print("single draws, z vs plain:", file=sys.stderr)
for r, s, c, b in D:
    if c not in ("plain", "dprime"):
        print(f"   {c:42s} {s:5.2f}  z={(s-pm)/psd:+.2f}", file=sys.stderr)
