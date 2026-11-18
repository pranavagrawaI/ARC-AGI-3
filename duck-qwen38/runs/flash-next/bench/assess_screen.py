"""Apply CARD.md's P+ selection rule to a downloaded speed-screen.json.

  python3 bench/assess_screen.py bench/out/speed-screen.json      # -> bench/out/screen-assessment.json
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

SUBSET_TOLERANCE = 0.03
GAIN = 1.10
MAX_ERROR_RATE = 0.01
MIN_PROMPT_MEAN = 15_000
MAX_TOKENIZE_FALLBACK = 0.05
FLAG_ENV = {"share": ("TAAF_VLLM_MTP_INDEX_SHARE_FOR_ITERATION", "1"),
            "prefix": ("TAAF_VLLM_ENABLE_PREFIX_CACHING", "1")}


def changes(overrides: dict) -> frozenset:
    out = {name for name, (key, value) in FLAG_ENV.items() if overrides.get(key) == value}
    if "TAAF_VLLM_MTP_TOKENS" in overrides:
        out.add(f"mtp{overrides['TAAF_VLLM_MTP_TOKENS']}")
    return frozenset(out)


def integrity(name: str, ev: dict, decisions: dict) -> list[str]:
    fails = []
    if not ev.get("ok"):
        return [f"did not run: {ev.get('error', '')[:200]}"]
    log, ch = ev.get("log") or {}, changes(ev.get("overrides") or {})
    if log.get("index_share_in_args") != ("share" in ch):
        fails.append(f"index_share_in_args={log.get('index_share_in_args')} but intended {'share' in ch}")
    if log.get("prefix_caching_in_args") != ("prefix" in ch):
        fails.append(f"prefix_caching_in_args={log.get('prefix_caching_in_args')} but intended {'prefix' in ch}")
    want_k = next((c[3:] for c in ch if c.startswith("mtp")), "3")
    if log.get("num_speculative_tokens") != [want_k]:
        fails.append(f"num_speculative_tokens={log.get('num_speculative_tokens')} but intended {want_k}")
    load = ev.get("load") or {}
    m0, m1 = load.get("measure_start"), load.get("ended")
    done = sum(w.get("client_completed", 0) for w in load.get("windows", []))
    errors = load.get("errors_after_warmup", 0)
    if not done or errors / (done + errors) > MAX_ERROR_RATE:
        fails.append(f"client errors {errors} vs completed {done}")
    if any("error" in w for w in load.get("windows", [])) or len(load.get("windows", [])) < 5:
        fails.append("missing measurement windows")
    if "prefix" in ch and not decisions.get("s2_equal"):
        fails.append(f"prefix-cached next-token logprobs diverge: {decisions.get('logprob_S2cached_vs_S1')} "
                     f"vs floors {decisions.get('logprob_floors')}")
    return fails


def assess(res: dict) -> dict:
    configs, decisions = res["configs"], res.get("decisions", {})
    rows = {}
    for name, ev in configs.items():
        wins = (ev.get("load") or {}).get("windows", [])
        gen = [w["gen_tok_s"] for w in wins if "gen_tok_s" in w]
        req = [w["requests_s"] for w in wins if "requests_s" in w]
        rows[name] = {"changes": sorted(changes(ev.get("overrides") or {})),
                      "integrity_failures": integrity(name, ev, decisions),
                      "gen_median": statistics.median(gen) if gen else None,
                      "req_median": statistics.median(req) if req else None, "gen_windows": gen,
                      "total": (ev.get("load") or {}).get("total")}
    s0 = rows.get("S0")
    if not s0 or s0["integrity_failures"] or s0["gen_median"] is None:
        return {"rows": rows, "decision": {"status": "INVALID", "p_plus": None, "reason": "S0 baseline invalid"}}
    # run-level validity (speed-screen-v2 card): the replay must look like P's live load
    for name, ev in configs.items():
        load = ev.get("load") or {}
        stats = load.get("stats") or {}
        counts = stats.get("tokenize_ok", 0) + stats.get("tokenize_fallback", 0)
        pt = (load.get("total") or {}).get("prompt_tokens_mean")
        rows[name]["prompt_tokens_mean"], rows[name]["tokenize_stats"] = pt, stats
        if ev.get("ok") and (pt is None or pt < MIN_PROMPT_MEAN):
            return {"rows": rows, "decision": {"status": "INVALID", "p_plus": None,
                                               "reason": f"{name} mean prompt {pt} < {MIN_PROMPT_MEAN}"}}
        if ev.get("ok") and counts and stats.get("tokenize_fallback", 0) / counts > MAX_TOKENIZE_FALLBACK:
            return {"rows": rows, "decision": {"status": "INVALID", "p_plus": None,
                                               "reason": f"{name} tokenize fallback {stats}"}}
    for r in rows.values():
        if r["gen_median"] is not None:
            r["gen_vs_S0"] = r["gen_median"] / s0["gen_median"]
            r["req_vs_S0"] = r["req_median"] / s0["req_median"]
    passing = {n: r for n, r in rows.items() if not r["integrity_failures"] and r["gen_median"] is not None}
    best = max(passing, key=lambda n: passing[n]["gen_median"])
    best_set = frozenset(passing[best]["changes"])
    subsets = [n for n, r in passing.items() if frozenset(r["changes"]) < best_set
               and r["gen_median"] >= (1 - SUBSET_TOLERANCE) * passing[best]["gen_median"]]
    choice = min(subsets, key=lambda n: (len(passing[n]["changes"]), -passing[n]["gen_median"])) if subsets else best
    c = rows[choice]
    above = sum(g > s0["gen_median"] for g in c["gen_windows"])
    ok = (choice != "S0" and c["gen_vs_S0"] >= GAIN and c["req_vs_S0"] >= GAIN and above >= 4)
    decision = {"status": "PASS" if ok else "FAIL", "best_by_gen": best, "p_plus": choice if ok else None,
                "p_plus_overrides": configs[choice]["overrides"] if ok else None,
                "gen_vs_S0": c["gen_vs_S0"], "req_vs_S0": c["req_vs_S0"], "windows_above_S0_median": above,
                "reason": ("P+ clears +10% on gen and requests with >=4/5 windows above S0" if ok else
                           f"{choice}: gen x{c['gen_vs_S0']:.3f}, req x{c['req_vs_S0']:.3f}, windows above {above}/5"
                           if choice != "S0" else "no config beats S0 by more than the subset tolerance")}
    return {"rows": rows, "decisions_in_notebook": decisions, "decision": decision}


if __name__ == "__main__":
    path = Path(sys.argv[1])
    out = assess(json.loads(path.read_text()))
    (path.parent / "screen-assessment.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({n: {k: r.get(k) for k in ("changes", "integrity_failures", "gen_median", "req_median",
                                               "gen_vs_S0", "req_vs_S0")} for n, r in out["rows"].items()}, indent=1))
    print(json.dumps(out["decision"], indent=1))
