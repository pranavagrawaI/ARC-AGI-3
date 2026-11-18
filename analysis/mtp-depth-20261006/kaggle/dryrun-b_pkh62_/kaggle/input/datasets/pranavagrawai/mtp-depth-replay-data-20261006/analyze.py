"""Pre-registered decision rule for the MTP-depth sweep (CARD.md), written before any result.

python3 -B analyze.py OUT_DIR    (reads OUT_DIR/sweep_state.json)

Baselines M3A, M3C, M3B run first, fourth and last (amended 2026-10-07, before any result).
Delta_i = U_m/U_3i - 1 for every measured baseline i; D = max pairwise |U_3i/U_3j - 1|;
W = max sub-window CV among the candidate and all baseline blocks.
Clean win: min_i Delta_i > D + W. A candidate that would clear W alone but not D + W is
"unresolved (launch drift)"; below W it is "no gain resolved". M3A and M3B are required; if
M3C failed the rule runs on A and B alone and says so.
SYSTEMATIC_DRIFT flags baselines monotonic in launch order (A < C < B or A > C > B). It is a
flag beside the verdict, not a change to it: 2 of 6 orderings are monotonic by chance.
Validity: launch M3A must reproduce production occupancy (mean R 8.97 +/- 0.40 and
P(R>=9) 0.72 +/- 0.10); otherwise the workload is invalid and nothing is decided.
"""
import json
import sys
from pathlib import Path

GATE_MEAN_R, GATE_MEAN_TOL = 8.97, 0.40
GATE_P9, GATE_P9_TOL = 0.72, 0.10
CACHE_DIAGNOSTIC_MIN = 0.90
CANDIDATES = ("M1", "M2", "M4")
BASELINES = ("M3A", "M3C", "M3B")  # launch order


def _summary(state, label):
    rec = state["launches"].get(label) or {}
    s = rec.get("summary") or {}
    return s if rec.get("status") == "ok" and s.get("U") else None


def decide(state):
    out = {"launches": {}, "candidates": {}, "cuts": state.get("cuts", []), "flags": [],
           "provenance": state.get("provenance")}
    for label, rec in state["launches"].items():
        s = rec.get("summary") or {}
        out["launches"][label] = {k: s.get(k) for k in (
            "U", "U_cv", "mean_R", "p_R_ge_9", "accept_len", "cached_fraction_block",
            "early_stop_rate", "errors", "warmup_criterion_met")}
        out["launches"][label]["status"] = rec.get("status")
    a, b = _summary(state, "M3A"), _summary(state, "M3B")
    if a is None:
        out["verdict"] = "INVALID: baseline M3A missing"
        return out
    gate_ok = (a["mean_R"] is not None and abs(a["mean_R"] - GATE_MEAN_R) <= GATE_MEAN_TOL
               and abs(a["p_R_ge_9"] - GATE_P9) <= GATE_P9_TOL)
    out["validity_gate"] = {"mean_R": a["mean_R"], "p_R_ge_9": a["p_R_ge_9"], "pass": gate_ok}
    out["cache_diagnostic"] = {"M3A_cached_fraction": a.get("cached_fraction_block"),
                               "below_0.90": (a.get("cached_fraction_block") or 0) < CACHE_DIAGNOSTIC_MIN}
    if not gate_ok:
        out["verdict"] = "INVALID WORKLOAD: M3A occupancy outside the production band; no refit"
        return out
    if b is None:
        out["verdict"] = "UNRESOLVED: baseline M3B missing, launch drift unmeasured"
        return out
    base = {lbl: s for lbl in BASELINES if (s := _summary(state, lbl)) is not None}
    out["baselines_in_launch_order"] = [{"label": lbl, "U": s["U"], "U_cv": s["U_cv"]}
                                        for lbl, s in base.items()]
    if "M3C" not in base:
        out["flags"].append("M3C_MISSING: drift from A and B only")
    us = [s["U"] for s in base.values()]
    D = max(us) / min(us) - 1  # the largest pairwise |U_i/U_j - 1|
    out["D"] = D
    if len(us) == 3 and (us[0] < us[1] < us[2] or us[0] > us[1] > us[2]):
        out["flags"].append(f"SYSTEMATIC_DRIFT: baselines monotonic in launch order, "
                            f"{(us[2] / us[0] - 1):+.2%} first to last")
    wins = []
    for m in CANDIDATES:
        c = _summary(state, m)
        if c is None:
            out["candidates"][m] = {"result": "not measured",
                                    "status": (state["launches"].get(m) or {}).get("status")}
            continue
        W = max(x for x in [c["U_cv"]] + [s["U_cv"] for s in base.values()] if x is not None)
        deltas = {lbl: c["U"] / s["U"] - 1 for lbl, s in base.items()}
        lo, hi = min(deltas.values()), max(deltas.values())
        if lo > D + W:
            result = "CLEAN WIN"
            wins.append((lo, m))
        elif lo > W:
            result = "UNRESOLVED (launch drift)"
        elif hi < -(D + W):
            result = "CLEAN LOSS"
        else:
            result = "NO GAIN RESOLVED"
        out["candidates"][m] = {"deltas": deltas, "W": W, "threshold": D + W,
                                "result": result, "early_stop_rate": c.get("early_stop_rate"),
                                "baseline_early_stop_rate": a.get("early_stop_rate")}
    out["verdict"] = (f"ADOPT CANDIDATE {max(wins)[1]} (lossless per Q7; next: short live "
                      f"harness qualification)") if wins else "KEEP M=3"
    return out


if __name__ == "__main__":
    state = json.loads((Path(sys.argv[1]) / "sweep_state.json").read_text())
    print(json.dumps(decide(state), indent=1))
