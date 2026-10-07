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
import json
import time
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

import verify_prelaunch

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
        response = client.kernels.kernels_api_client.get_accelerator_quota_statistics()
    g = getattr(response, "gpu_quota", None)
    used, reserved = getattr(g, "time_used", None), getattr(g, "time_reserved", None)
    if not isinstance(used, dt.timedelta) or not isinstance(reserved, dt.timedelta):
        raise ValueError("GPU quota usage/reservation missing; cannot admit a launch")
    used_s, reserved_s = used.total_seconds(), reserved.total_seconds()
    if used_s < 0 or reserved_s < 0:
        raise ValueError("GPU quota duration is negative")
    # The SDK string serializer emits e.g. 21783.604.0s and discards duration days.
    # Use its parsed timedelta objects for admission; retain the raw serialization too.
    q = json.loads(str(response))
    q["gpuQuotaSeconds"] = {"timeUsed": used_s, "timeReserved": reserved_s}
    return used_s + reserved_s, q


def active_sessions(api):
    owner = json.loads((NOTEBOOK_DIR / "kernel-metadata.json").read_text())["id"].split("/")[0]
    kernels = api.kernels_list(mine=True, user=owner, sort_by="dateRun", page_size=20)
    active = []
    for k in kernels:
        ref = getattr(k, "ref", None)
        if not ref:
            # Kaggle returns redacted historical records with no runnable identity.
            # Match the observed sentinel exactly; any other missing identity blocks launch.
            stamp = getattr(k, "last_run_time", None)
            if (getattr(k, "title", None) == "[Private Notebook]"
                    and not getattr(k, "author", None)
                    and isinstance(stamp, dt.datetime)
                    and stamp.replace(tzinfo=dt.timezone.utc) == dt.datetime(2010, 4, 1, tzinfo=dt.timezone.utc)
                    and not getattr(k, "current_version_number", None)):
                continue
            active.append((None, "unknown (missing kernel ref)"))
            continue
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


def identity():
    return {**verify_prelaunch.verify(), "hard_timeout_s": HARD_TIMEOUT_S}


def priority_resolved():
    if GATE.exists():
        decision = json.loads(GATE.read_text())
        if decision.get("decision") == "hold":
            return True
        if decision.get("decision") == "launch":
            receipt = GATE.parent / "control/launch-response.json"
            return receipt.exists() and bool(json.loads(receipt.read_text()).get("versionNumber"))
    if WAVE2_RESOLVED.exists():
        decision = json.loads(WAVE2_RESOLVED.read_text())
        readout = (HERE / decision["readout_json"]).resolve()
        return (decision["decision"] == "wave-2 control priority resolved: HOLD, control will not launch"
                and verify_prelaunch.build_notebook.sha256(readout) == decision["readout_sha256"])
    return False


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
    ident = identity()  # prove the tag, payload, notebook and uploaded readback before polling
    import requests
    original_send = requests.Session.send
    def bounded_send(session, request, **kwargs):
        kwargs.setdefault("timeout", (15, 45))
        return original_send(session, request, **kwargs)
    requests.Session.send = bounded_send
    api = KaggleApi()
    api.authenticate()
    until = time.time() + args.hours * 3600  # wall clock: monotonic stops while the laptop sleeps
    while time.time() < until:
        reasons = []
        try:
            if not priority_resolved():
                reasons.append("wave-2 control priority unresolved")
            refresh_token()
            api = KaggleApi()
            api.authenticate()
            active = active_sessions(api)
            used, q = quota_used_s(api)
        except Exception as e:  # transient API/network failure: never launch blind, keep waiting
            log(f"poll error {type(e).__name__}: {str(e)[:200]}")
            if args.dry:
                raise SystemExit("dry check unavailable; no launch")
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
        ident = identity()  # recheck after any wait; never launch a changed payload
        save(EVID / "launch-identity.json", ident)
        save(EVID / "quota-before.json", {"utc": now(), "quota": q, "active_before": active})
        attempt = {"utc": now(), "state": "attempting", "identity": ident}
        try:
            with attempt_path.open("x") as f:  # a second watcher cannot consume the same authority
                json.dump(attempt, f, indent=2)
                f.write("\n")
        except FileExistsError:
            raise SystemExit("launch already claimed by another watcher; do not retry")
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
