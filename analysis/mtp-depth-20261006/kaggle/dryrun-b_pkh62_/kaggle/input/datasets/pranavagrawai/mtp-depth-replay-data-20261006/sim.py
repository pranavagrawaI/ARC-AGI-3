"""Closed-loop discrete-event model of the control run from the minute-40 checkpoint; fits alpha.

python3 -B sim.py [workload.json]    (fits alpha, writes alpha.json)

Server: per-stream decode rate r(R) = time-weighted gen tok/s at R running / R, from serve.log
(this already folds in the average prefill interleave), plus a serialized prefill of each
request's uncached tokens at the median large-batch prefill rate.
Gate: S slots. A game holds its slot across tool gaps. On a trimmed request (prefix break)
it hands the slot over and re-queues, as dfranzen's _PriorityGate does. Waiting games are
admitted in the order production admitted them (estimated send time of their pending
request), so the replay follows production's realized admission sequence.
Gaps: for turns without a handover, turn interval minus modelled server time, split evenly
over the turn's requests; handover turns contain gate waits and use the game's median gap.
alpha scales every gap; fitted so the M=3/S=10 model reproduces production's mean running
count over run-minutes 30-60 (8.97), then frozen.
"""
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).parent
WORKLOAD = HERE / "workload.json"
CHECKPOINT_MIN = 40.0
SLOTS = 10
PREFILL_TOK_S = 11942.0  # median input tok/s of prefill batches >= 4096 new tokens
# time-weighted gen tok/s by running count over the whole control serve.log, M=3.
# (Per-line medians overweight fast intervals: a log line covers a fixed number of decode
# steps, not a fixed time. They gave 756/789 at R=9/10.)
GEN_TOK_S = {1: 175, 2: 285, 3: 316, 4: 532, 5: 495, 6: 558, 7: 660, 8: 669, 9: 687, 10: 704}
TARGET_MEAN_R, TARGET_P9 = 8.97, 0.72
DT = 0.05
HORIZON_S, MEASURE_FROM_S = 20 * 60, 3 * 60


def per_stream_rate(r):
    return GEN_TOK_S[min(max(r, 1), SLOTS)] / max(r, 1)


R_PROD = per_stream_rate(9)


def server_time(p):
    return max(0, p["prompt_tokens"] - p["cached_tokens"]) / PREFILL_TOK_S + \
        p["completion_tokens"] / R_PROD


def turns(reqs):
    out, cur = [], []
    for i, p in enumerate(reqs):
        if cur and p["analysis_step"] != reqs[cur[-1]]["analysis_step"]:
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return out


def trim_position(reqs, idx):
    """Position within the turn of the first trimmed request (gate handover), else None.

    dfranzen trims and may hand over before every request of a turn, so the gate wait
    sits in front of that request, after the turn's header timestamp."""
    return next((n for n, i in enumerate(idx) if i > 0 and not reqs[i]["extends_previous"]), None)


def annotate(reqs):
    """Per-request gap estimate and estimated send time (minutes)."""
    ts = turns(reqs)
    for p in reqs:
        p["gap"] = None
    for k, idx in enumerate(ts):
        t_next = reqs[ts[k + 1][0]]["t_min"] if k + 1 < len(ts) else None
        if trim_position(reqs, idx) is not None or t_next is None or reqs[idx[0]]["t_min"] is None:
            continue
        resid = (t_next - reqs[idx[0]]["t_min"]) * 60 - sum(server_time(reqs[i]) for i in idx)
        for i in idx:
            reqs[i]["gap"] = max(0.0, resid / len(idx))
    known = [p["gap"] for p in reqs if p["gap"] is not None]
    med = st.median(known) if known else 5.0
    for p in reqs:
        p["gap_known"] = p["gap"] is not None
        if p["gap"] is None:
            p["gap"] = med
    for k, idx in enumerate(ts):
        t0 = reqs[idx[0]]["t_min"]
        t_next = reqs[ts[k + 1][0]]["t_min"] if k + 1 < len(ts) else None
        cut = trim_position(reqs, idx)
        if cut is None or t_next is None:
            cut = len(idx)
        t = t0 * 60
        for i in idx[:cut]:
            reqs[i]["send_min"] = t / 60
            t += server_time(reqs[i]) + reqs[i]["gap"]
        t_fwd = t
        t = t_next * 60 if cut < len(idx) else t
        for i in reversed(idx[cut:]):
            t -= server_time(reqs[i]) + reqs[i]["gap"]
            reqs[i]["send_min"] = max(t, t_fwd) / 60
    return med


def build(workload):
    games = {}
    for g, d in workload["games"].items():
        reqs = d["requests"]
        med = annotate(reqs)
        ck = next((i for i, p in enumerate(reqs) if p["send_min"] >= CHECKPOINT_MIN), None)
        if ck is None:
            continue
        pending_trim = not reqs[ck]["extends_previous"]
        games[g] = dict(reqs=reqs[ck:], median_gap=med, waiting_at_ck=pending_trim,
                        ck_send=reqs[ck]["send_min"])
    return games


def initial_admission(games, slots=SLOTS):
    """Slot holders and FIFO wait queue at the checkpoint, in production's order."""
    order = sorted(games, key=lambda g: games[g]["ck_send"])
    holders = [g for g in order if not games[g]["waiting_at_ck"]][:slots]
    return holders, [g for g in order if g not in holders]


def simulate(games, alpha, slots=SLOTS, horizon=HORIZON_S):
    holders, queue = initial_admission(games, slots)
    st_ = {g: dict(i=0, phase="idle", rem=0.0) for g in games}
    prefill_q, done = [], set()

    def start_request(g):
        p = games[g]["reqs"][st_[g]["i"]]
        st_[g].update(phase="prefill_wait", pre=max(0, p["prompt_tokens"] - p["cached_tokens"]),
                      rem=float(p["completion_tokens"]))
        prefill_q.append(g)

    def admit():
        while queue and len(holding) < slots:
            g = queue.pop(0)
            holding.add(g)
            start_request(g)

    holding = set(holders)
    for g in holders:
        start_request(g)
    admit()
    t, trace, handovers = 0.0, [], 0
    while t < horizon:
        # prefill: serialized, head of queue
        if prefill_q:
            g = prefill_q[0]
            st_[g]["phase"] = "prefill"
            st_[g]["pre"] -= PREFILL_TOK_S * DT
            if st_[g]["pre"] <= 0:
                prefill_q.pop(0)
                st_[g]["phase"] = "decode"
        decoding = [g for g in holding if st_[g]["phase"] == "decode"]
        r = len(decoding)
        rate = per_stream_rate(r) if r else 0.0
        for g in decoding:
            st_[g]["rem"] -= rate * DT
            if st_[g]["rem"] <= 0:
                p = games[g]["reqs"][st_[g]["i"]]
                st_[g].update(phase="gap", rem=alpha * p["gap"])
        for g in list(holding):
            s = st_[g]
            if s["phase"] != "gap":
                continue
            s["rem"] -= DT
            if s["rem"] > 0:
                continue
            s["i"] += 1
            reqs = games[g]["reqs"]
            if s["i"] >= len(reqs):
                holding.discard(g)
                done.add(g)
                s["phase"] = "done"
            elif not reqs[s["i"]]["extends_previous"]:
                holding.discard(g)
                queue.append(g)
                s["phase"] = "waiting"
                handovers += 1
            else:
                start_request(g)
        admit()
        trace.append((t, r, len(holding)))
        t += DT
    win = [x for x in trace if x[0] >= MEASURE_FROM_S]
    mean_r = sum(x[1] for x in win) / len(win)
    p9 = sum(x[1] >= 9 for x in win) / len(win)
    return dict(alpha=alpha, mean_R=round(mean_r, 3), p_R_ge_9=round(p9, 3),
                mean_holding=round(sum(x[2] for x in win) / len(win), 2),
                games_done=len(done), handovers=handovers)


def main():
    workload = json.loads((Path(sys.argv[1]) if len(sys.argv) > 1 else WORKLOAD).read_text())
    games = build(workload)
    known = sum(p["gap_known"] for g in games.values() for p in g["reqs"])
    total = sum(len(g["reqs"]) for g in games.values())
    gaps = [p["gap"] for g in games.values() for p in g["reqs"] if p["gap_known"]]
    print(f"games at checkpoint: {len(games)}; waiting at checkpoint (pending trim): "
          f"{sum(g['waiting_at_ck'] for g in games.values())}; gaps known {known}/{total}; "
          f"gap median {st.median(gaps):.1f}s p75 {st.quantiles(gaps, n=4)[2]:.1f}s; r_prod {R_PROD:.1f} tok/s")
    results = [simulate(games, a) for a in
               (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)]
    for res in results:
        print(res)
    best = min(results, key=lambda x: abs(x["mean_R"] - TARGET_MEAN_R))
    lo, hi = sorted(results, key=lambda x: abs(x["mean_R"] - TARGET_MEAN_R))[:2], None
    # refine by bisection between the two grid points bracketing the target
    a_lo = max((x for x in results if x["mean_R"] >= TARGET_MEAN_R), key=lambda x: x["alpha"], default=None)
    a_hi = min((x for x in results if x["mean_R"] < TARGET_MEAN_R), key=lambda x: x["alpha"], default=None)
    if a_lo and a_hi:
        lo_a, hi_a = a_lo["alpha"], a_hi["alpha"]
        for _ in range(8):
            mid = (lo_a + hi_a) / 2
            res = simulate(games, mid)
            if res["mean_R"] >= TARGET_MEAN_R:
                lo_a = mid
            else:
                hi_a = mid
        best = simulate(games, round((lo_a + hi_a) / 2, 3))
    print("FIT", best, "target", dict(mean_R=TARGET_MEAN_R, p_R_ge_9=TARGET_P9))
    (HERE / "alpha.json").write_text(json.dumps(dict(
        best, target_mean_R=TARGET_MEAN_R, target_p_R_ge_9=TARGET_P9,
        checkpoint_min=CHECKPOINT_MIN, slots=SLOTS, prefill_tok_s=PREFILL_TOK_S,
        gen_tok_s=GEN_TOK_S, horizon_s=HORIZON_S, measure_from_s=MEASURE_FROM_S), indent=1))


if __name__ == "__main__":
    main()
