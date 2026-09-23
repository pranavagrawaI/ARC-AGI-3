"""Read an arm of qwen38-fixes against the card. usage: readback.py <arm output dir> [baseline output dir]
Prints: levels (and with the top game dropped), games with a level, yielded-turn share, ls20 actions,
median s/turn, game-over retries, prompt-fix uptake, MTP acceptance from the vLLM log."""
import glob, json, os, re, statistics as st, sys
from datetime import datetime

def arm(d):
    art = os.path.join(d, "artifacts"); per = {}; turns = {"act": 0, "yield": 0, "err": 0}; durs = []
    levelup_review = retry_after_go = 0
    for f in sorted(glob.glob(art + "/*events.jsonl")):
        g = os.path.basename(f)[:4]; ev = [json.loads(l) for l in open(f)]
        acts = [e for e in ev if e["type"] == "action"]; an = [e for e in ev if e["type"] == "analysis"]
        lv = sum(str(a.get("level_completed")) == "True" for a in acts)
        gos = [i for i, a in enumerate(acts) if str(a.get("game_over")) == "True"]
        per[g] = {"levels": lv, "actions": len(acts), "turns": len(an), "game_overs": len(gos),
                  "levels_after_go": sum(1 for i, a in enumerate(acts) if str(a.get("level_completed")) == "True" and gos and i > gos[0])}
        ts = []
        for e in an:
            t = e["transcript"]; tail = t[t.rfind("[ANALYZER STATUS]"):]
            k = "act" if "step_executed: True" in tail else ("yield" if "turn_time_budget" in tail else "err"); turns[k] += 1
            m = re.search(r"\| (\d\d:\d\d:\d\d) \|", t[:200]); ts.append(datetime.strptime(m.group(1), "%H:%M:%S") if m else None)
            if "Cross-level notes" in t and "verified rule" in t: levelup_review += 1
        durs += [(b - a).total_seconds() for a, b in zip(ts, ts[1:]) if a and b and b > a]
    return per, turns, durs

def mtp(d):
    log = os.path.join(d, "vllm-openai-server.log")
    if not os.path.exists(log): return "no vllm log"
    txt = open(log, errors="ignore").read()
    spec = re.findall(r"speculative_config=([^,]+(?:,[^=]+)?)", txt)[:1]
    acc = [float(x) for x in re.findall(r"[Aa]vg(?:erage)? [Dd]raft acceptance rate: ([0-9.]+)%?", txt)]
    mean_len = [float(x) for x in re.findall(r"[Mm]ean acceptance length: ([0-9.]+)", txt)]
    gen = [float(x) for x in re.findall(r"Avg generation throughput: ([0-9.]+) tokens/s", txt)]
    return {"speculative_config": spec, "acceptance_rate_median": st.median(acc) if acc else None,
            "mean_acceptance_length_median": st.median(mean_len) if mean_len else None,
            "gen_tok_s_median": st.median([g for g in gen if g > 0]) if gen else None}

def show(label, d):
    per, turns, durs = arm(d); n = sum(turns.values()) or 1
    lv = sorted((v["levels"] for v in per.values()), reverse=True)
    print(f"== {label}: {d}")
    print(f"levels {sum(lv)} (top game dropped {sum(lv[1:])}), games with a level {sum(x > 0 for x in lv)}/{len(per)}")
    print(f"turns {n}: acted {turns['act']} ({100*turns['act']/n:.0f}%), yielded {turns['yield']} ({100*turns['yield']/n:.0f}%), error {turns['err']}")
    print(f"median s/turn {st.median(durs) if durs else None}; ls20 actions {per.get('ls20', {}).get('actions')}")
    print(f"game overs {sum(v['game_overs'] for v in per.values())} in {sum(v['game_overs'] > 0 for v in per.values())} games; levels after a game over {sum(v['levels_after_go'] for v in per.values())}")
    print("mtp:", mtp(d))
    return per

a = show("arm", sys.argv[1])
if len(sys.argv) > 2:
    b = show("baseline", sys.argv[2])
    print("per game (arm - baseline):", {g: a[g]["levels"] - b.get(g, {}).get("levels", 0) for g in sorted(a) if a[g]["levels"] != b.get(g, {}).get("levels", 0)})
