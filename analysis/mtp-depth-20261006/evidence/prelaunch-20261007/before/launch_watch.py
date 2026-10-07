"""Launch the MTP-depth sweep once a GPU slot is free (CARD.md priority rule). One attempt, no retry.

~/.local/share/uv/tools/kaggle/bin/python -B launch_watch.py [--hours 60] [--dry]

Waits until (1) wave-2's control has priority resolved: its gate has decided, or
evidence/wave2-priority-resolved.json records the card's HOLD from collected outputs (its gate
can wait forever on a hung monitor), (2) fewer than two of the account's kernels are RUNNING/QUEUED (Kaggle refuses a
push at the cap, and a refused push can poison a kernel id), and (3) quota leaves room:
API timeUsed + this run's hard cap <= 30 h (the Kaggle UI cap; the API's totalTimeAllowed is
wrong). Then pushes kaggle/notebook with a hard session timeout and records everything under
evidence/. An existing launch-attempt.json means do not push again.
"""
import argparse
import datetime as dt
import hashlib
import json
import time
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

HERE = Path(__file__).resolve().parent
EVID = HERE / "evidence"
NOTEBOOK_DIR = HERE / "kaggle/notebook"
GATE = HERE / "../../solvers/tycho_dfranzen/experiment/wave-2/control-gate.json"
WAVE2_RESOLVED = EVID / "wave2-priority-resolved.json"
TAG = "mtp-depth-sweep-prelaunch-20261007"
WEEKLY_CAP_S = 30 * 3600
HARD_TIMEOUT_S = 4 * 3600 + 30 * 60  # notebook self-caps at 4 h
ACTIVE = ("RUNNING", "QUEUED")


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def log(msg):
    line = f"{now()} {msg}"
    print(line, flush=True)
    with (EVID / "launch-watch.log").open("a") as f:
        f.write(line + "\n")


def quota_used_s(api):
    with api.build_kaggle_client() as client:
        q = json.loads(str(client.kernels.kernels_api_client.get_accelerator_quota_statistics()))
    g = q.get("gpuQuota") or q.get("gpu_quota") or {}
    parse = lambda s: float(str(s or "0").rstrip("s").split(".")[0] or 0)
    return parse(g.get("timeUsed")) + parse(g.get("timeReserved")), q


def active_sessions(api):
    kernels = api.kernels_list(mine=True, sort_by="dateRun", page_size=20)
    active = []
    for k in kernels:
        ref = getattr(k, "ref", None) or getattr(k, "id", None)
        try:
            status = str(api.kernels_status(ref).status)
        except Exception as e:
            status = f"unknown ({type(e).__name__})"
        if status.startswith("unknown") or any(a in status for a in ACTIVE):
            active.append((ref, status))
    return active


def refresh_token():
    """The OAuth access token lives 12 h and the SDK refreshes only 30 min after expiry."""
    from kagglesdk import KaggleClient
    from kagglesdk.kaggle_creds import KaggleCredentials
    creds = KaggleCredentials.load(KaggleClient())
    exp = creds._access_token_expiration
    if exp is None or exp - dt.datetime.now(dt.timezone.utc) < dt.timedelta(hours=1):
        creds.refresh_access_token()
        log(f"refreshed Kaggle token (was expiring {exp})")


def git(*a):
    import subprocess
    return subprocess.run(["git", *a], cwd=HERE, capture_output=True, text=True, check=True).stdout.strip()


def identity():
    nb = next(NOTEBOOK_DIR.glob("*.ipynb"))
    return {"prelaunch_tag": TAG, "tag_commit": git("rev-list", "-n1", TAG),
            "provenance": json.loads((HERE / "kaggle/dataset/provenance.json").read_text()),
            "notebook": nb.name, "notebook_sha256": hashlib.sha256(nb.read_bytes()).hexdigest(),
            "kernel_metadata": json.loads((NOTEBOOK_DIR / "kernel-metadata.json").read_text()),
            "dataset_manifest_sha256": hashlib.sha256((HERE / "kaggle/dataset/manifest.json").read_bytes()).hexdigest(),
            "alpha": json.loads((HERE / "alpha.json").read_text())["alpha"],
            "hard_timeout_s": HARD_TIMEOUT_S}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=60)
    ap.add_argument("--poll-s", type=float, default=120)
    ap.add_argument("--dry", action="store_true", help="check conditions once, never push")
    args = ap.parse_args()
    EVID.mkdir(exist_ok=True)
    attempt_path = EVID / "launch-attempt.json"
    if attempt_path.exists():
        raise SystemExit("launch already attempted; inspect evidence/, do not retry")
    if git("status", "--porcelain", "--", ".", ":!evidence/launch-watch.log"):
        raise SystemExit("working tree differs from the committed pre-launch state; commit first")
    api = KaggleApi()
    api.authenticate()
    until = time.time() + args.hours * 3600  # wall clock: monotonic stops while the laptop sleeps
    while time.time() < until:
        reasons = []
        if not GATE.exists() and not WAVE2_RESOLVED.exists():
            reasons.append("wave-2 control gate undecided")
        try:
            refresh_token()
            api = KaggleApi()
            api.authenticate()
            active = active_sessions(api)
            used, q = quota_used_s(api)
        except Exception as e:  # transient API/network failure: never launch blind, keep waiting
            log(f"poll error {type(e).__name__}: {str(e)[:200]}")
            time.sleep(args.poll_s)
            continue
        if any(s.startswith("unknown") for _, s in active) or len(active) >= 2:
            reasons.append(f"slots full or unknown: {active}")
        if used + HARD_TIMEOUT_S > WEEKLY_CAP_S:
            reasons.append(f"quota: used {used / 3600:.2f} h + cap {HARD_TIMEOUT_S / 3600:.1f} h > 30 h")
        if args.dry:
            log(f"dry check: {'READY' if not reasons else '; '.join(reasons)}")
            return
        if reasons:
            log("waiting: " + "; ".join(reasons))
            time.sleep(args.poll_s)
            continue
        ident = identity()
        save(EVID / "launch-identity.json", ident)
        save(EVID / "quota-before.json", {"utc": now(), "quota": q, "active_before": active})
        attempt = {"utc": now(), "state": "attempting", "identity": ident}
        save(attempt_path, attempt)
        try:
            result = json.loads(str(api.kernels_push(str(NOTEBOOK_DIR), timeout=str(HARD_TIMEOUT_S))))
            save(EVID / "launch-response.json", result)
            if result.get("error") or not result.get("versionNumber"):
                raise RuntimeError(f"push not accepted: {result}")
            attempt.update(state="accepted", response=result)
            log(f"LAUNCHED {result}")
        except Exception as e:
            attempt.update(state="acceptance unresolved; inspect, do not retry", error=repr(e)[:500])
            log(f"PUSH FAILED {e!r}")
            raise
        finally:
            save(attempt_path, attempt)
        return
    log("watch window expired without launching")


if __name__ == "__main__":
    main()
