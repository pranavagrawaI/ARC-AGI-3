"""One-attempt launcher for an oracle-hint session. Admission: identity unchanged since build, a free GPU slot,
weekly quota, and the 8 GPU-h experiment ceiling (actual time of finished sessions, hard caps of unfinished ones).
A launch-attempt.json in the session directory means never push it again.

  ~/.local/share/uv/tools/kaggle/bin/python -B launch.py a0|hints [--dry]
"""
import datetime as dt, hashlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mtp-depth-20261006"))
import launch_watch as lw  # quota_used_s, active_sessions, refresh_token (field-tested)

CEILING_S = 8 * 3600
WEEKLY_CAP_S = 30 * 3600
HARD_CAP_S = 4 * 3600
LEDGER = HERE / "budget.json"


def now(): return dt.datetime.now(dt.timezone.utc).isoformat()


def save(p, v):
    t = p.with_suffix(p.suffix + ".tmp"); t.write_text(json.dumps(v, indent=2) + "\n"); t.replace(p)


def committed_s(ledger):
    return sum(r["actual_s"] if r.get("actual_s") is not None else r["hard_cap_s"] for r in ledger["runs"])


def verify(session):
    d = HERE / "sessions" / session
    ident = json.loads((d / "identity.json").read_text())
    nb = d / "kernel" / f"{ident['slug']}.ipynb"
    assert hashlib.sha256(nb.read_bytes()).hexdigest() == ident["notebook_sha256"], "notebook drifted since build"
    m = json.loads((d / "kernel" / "kernel-metadata.json").read_text())
    assert m == ident["metadata"] and m["is_private"] and m["machine_shape"] == "NvidiaRtxPro6000"
    assert m["competition_sources"] == ["arc-prize-2026-arc-agi-3"]
    if (d / "launch-attempt.json").exists():
        raise SystemExit(f"{session}: launch already attempted; do not retry")
    return {**ident, "card_sha256": hashlib.sha256((HERE / "EXPERIMENT-CARD.md").read_bytes()).hexdigest()}


def main():
    session, dry = sys.argv[1], "--dry" in sys.argv
    ident = verify(session)
    d = HERE / "sessions" / session
    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {"ceiling_s": CEILING_S, "runs": []}
    lw.EVID = HERE
    lw.refresh_token()
    api = lw.KaggleApi(); api.authenticate()
    lw.NOTEBOOK_DIR = d / "kernel"
    active = lw.active_sessions(api)
    used, q = lw.quota_used_s(api)
    reasons = []
    if any(str(s).startswith("unknown") for _, s in active) or len(active) + 1 > 2:
        reasons.append(f"slots: active {active}")
    if used + HARD_CAP_S > WEEKLY_CAP_S:
        reasons.append(f"weekly quota: used {used/3600:.2f} h + {HARD_CAP_S/3600:.1f} h > 30 h")
    if committed_s(ledger) + HARD_CAP_S > CEILING_S:
        reasons.append(f"experiment ceiling: committed {committed_s(ledger)/3600:.2f} h + {HARD_CAP_S/3600:.1f} h > 8 h")
    print(f"{now()} {session}: quota used {used/3600:.2f} h, active {active}, committed {committed_s(ledger)/3600:.2f} h:",
          "READY" if not reasons else "; ".join(reasons))
    if dry or reasons:
        return
    attempt = {"utc": now(), "state": "attempting", "identity": ident, "hard_cap_s": HARD_CAP_S,
               "quota_before": q, "active_before": active}
    with (d / "launch-attempt.json").open("x") as f:
        json.dump(attempt, f, indent=2)
    ledger["runs"].append({"run": session, "hard_cap_s": HARD_CAP_S, "actual_s": None, "launched_utc": now()})
    save(LEDGER, ledger)  # reserve before pushing: a lost response still counts
    try:
        result = json.loads(str(api.kernels_push(str(d / "kernel"), timeout=str(HARD_CAP_S))))
        save(d / "launch-response.json", result)
        if result.get("error") or not result.get("versionNumber"):
            raise RuntimeError(f"push not accepted: {result}")
        attempt.update(state="accepted", response=result)
        print(f"{now()} LAUNCHED {session} {result}")
    except Exception as e:
        attempt.update(state="acceptance unresolved; inspect, do not retry", error=repr(e)[:500])
        print(f"{now()} PUSH FAILED {session} {e!r}")
    save(d / "launch-attempt.json", attempt)


if __name__ == "__main__":
    main()
