"""Six-launch MTP-depth sweep orchestrator (CARD.md), resumable.

Local rehearsal against the mock server:
    python3 -B sweep.py --mock --out /tmp/mtp-rehearsal --time-scale 20

In Kaggle the notebook builds a ServerFactory around the incumbent launcher and calls
run_sweep(...) directly. One launch = start server -> wait /health -> replay.run ->
summarize -> stop server. A launch that does not come up is recorded and skipped, never
retried. Completed launches are skipped on rerun (sweep_state.json).
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import analyze
import replay
import sim

HERE = Path(__file__).parent
PLAN = [  # label, speculative_num_steps, baseline?
    ("M3A", 3, True), ("M1", 1, False), ("M2", 2, False), ("M3C", 3, True), ("M4", 4, False),
    ("M3B", 3, True)]
PROFILE = ("PROFILE_M3", 3)
STARTUP_EST_S = 9 * 60       # control run: 8.4 min to /health
HEALTH_TIMEOUT_S = 20 * 60
MEASURE_S, MEASURE_SHORT_S = 240.0, 180.0
WARMUP_MAX_S = 600.0
PROFILE_S = 60.0


class MockFactory:
    """Starts mock_server.py; speed matches the replay time_scale."""

    def __init__(self, port, speed, fail_steps=(), speed_mult=None, error_rate=0.0):
        self.port, self.speed, self.fail_steps = port, speed, set(fail_steps)
        self.speed_mult = speed_mult or {}  # planted per-M effects for positive controls
        self.error_rate = error_rate

    def start(self, steps, nsys=False, log_path=None):
        cmd = [sys.executable, "-B", str(HERE / "mock_server.py"), str(self.port),
               "--speed", str(self.speed * self.speed_mult.get(steps, 1.0)), "--load-seconds", "1",
               "--error-rate", str(self.error_rate)]
        if steps in self.fail_steps:
            cmd.append("--fail-start")
        log = open(log_path or os.devnull, "ab")
        return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=HERE,
                                start_new_session=True)

    def stop(self, proc):
        stop_process_group(proc)

    def profile_hooks(self, proc):
        return (lambda: True), (lambda: True)


def stop_process_group(proc, grace=30):
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=grace)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=grace)


def wait_health(proc, base_url, timeout):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if proc.poll() is not None:
            return False, f"server exited with {proc.returncode}"
        try:
            with urllib.request.urlopen(base_url + "/health", timeout=5) as r:
                if r.status == 200:
                    return True, f"ready after {time.monotonic() - t0:.0f}s"
        except Exception:
            pass
        time.sleep(2)
    return False, "health timeout"


def plan_for_time(remaining_s, done, profile_mode, time_scale):
    """Cut profile, shorten non-baseline blocks, then drop M4 and M2.
    All three baselines and M1 are protected. A missed baseline makes analysis unresolved."""
    per = lambda measure: (STARTUP_EST_S + WARMUP_MAX_S + measure + 60) / time_scale
    todo = [(lbl, st, base) for lbl, st, base in PLAN if lbl not in done]
    with_profile = profile_mode == "separate" and PROFILE[0] not in done
    cuts, measure_nb = [], MEASURE_S

    def need():
        n = sum(per(MEASURE_S if base else measure_nb) for _, _, base in todo)
        return n + (per(PROFILE_S) if with_profile else 0)

    if need() > remaining_s and with_profile:
        with_profile = False
        cuts.append("profile")
    if need() > remaining_s:
        measure_nb = MEASURE_SHORT_S
        cuts.append(f"non-baseline blocks {MEASURE_S:.0f}->{MEASURE_SHORT_S:.0f}s")
    for lbl in ("M4", "M2"):
        if need() > remaining_s and any(t[0] == lbl for t in todo):
            todo = [t for t in todo if t[0] != lbl]
            cuts.append(f"drop {lbl}")
    return todo, with_profile, measure_nb, cuts


def run_launch(label, steps, *, factory, base_url, out, workload, logs_dir, alpha, model,
               measure_s, time_scale, nsys=False, profile_s=None, provenance=None):
    d = out / label
    d.mkdir(parents=True, exist_ok=True)
    rec = {"label": label, "speculative_num_steps": steps, "nsys": nsys, "started": time.time()}
    proc = None
    try:
        proc = factory.start(steps, nsys=nsys, log_path=d / "server.log")
        ok, why = wait_health(proc, base_url, HEALTH_TIMEOUT_S / time_scale)
        rec["health"] = why
        if not ok:
            rec["status"] = "launch_failed"
            return rec
        profile = None
        if profile_s:
            start_fn, stop_fn = factory.profile_hooks(proc)
            profile = (start_fn, stop_fn, profile_s)
        r = replay.Replay(workload=workload, logs_dir=logs_dir, alpha=alpha, base_url=base_url,
                          model=model, events_path=d / "events.jsonl", time_scale=time_scale,
                          header={"label": label, "speculative_num_steps": steps, "nsys": nsys,
                                  **(provenance or {})})
        r.run(measure_s=measure_s, warmup_max_s=WARMUP_MAX_S, profile=profile)
        rec["summary"] = replay.summarize(d / "events.jsonl")
        rec["status"] = "ok" if rec["summary"].get("valid") else "invalid"
        return rec
    except Exception as e:
        rec["status"] = "error"
        rec["error"] = repr(e)[:1000]
        return rec
    finally:
        if proc is not None:
            factory.stop(proc)
        rec["finished"] = time.time()
        (d / "launch.json").write_text(json.dumps(rec, indent=1))


def run_sweep(*, factory, base_url, out, workload, logs_dir, alpha, model,
              deadline_monotonic, profile_mode="separate", time_scale=1.0, provenance=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "sweep_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"launches": {}}
    state["provenance"] = provenance or {}
    done = set(state["launches"])
    todo, with_profile, measure_nb, cuts = plan_for_time(
        deadline_monotonic - time.monotonic(), done, profile_mode, time_scale)
    state.setdefault("cuts", []).extend(c for c in cuts if c not in state["cuts"])
    state_path.write_text(json.dumps(state, indent=1))
    for label, steps, base in todo:
        if time.monotonic() >= deadline_monotonic:
            state["launches"][label] = {"label": label, "status": "skipped_deadline"}
            continue
        rec = run_launch(label, steps, factory=factory, base_url=base_url, out=out,
                         workload=workload, logs_dir=logs_dir, alpha=alpha, model=model,
                         measure_s=MEASURE_S if base else measure_nb, time_scale=time_scale,
                         provenance=provenance)
        state["launches"][label] = rec
        state_path.write_text(json.dumps(state, indent=1))
    if with_profile and time.monotonic() < deadline_monotonic:
        rec = run_launch(PROFILE[0], PROFILE[1], factory=factory, base_url=base_url, out=out,
                         workload=workload, logs_dir=logs_dir, alpha=alpha, model=model,
                         measure_s=MEASURE_SHORT_S, time_scale=time_scale, nsys=True,
                         profile_s=PROFILE_S, provenance=provenance)
        state["launches"][PROFILE[0]] = rec
        state_path.write_text(json.dumps(state, indent=1))
    verdict = analyze.decide(state)
    (out / "verdict.json").write_text(json.dumps(verdict, indent=1))
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--out", default="/tmp/mtp-rehearsal")
    ap.add_argument("--time-scale", type=float, default=20.0)
    ap.add_argument("--port", type=int, default=18001)
    ap.add_argument("--budget-s", type=float, default=None, help="session budget (unscaled)")
    ap.add_argument("--fail-steps", default="", help="comma list of M values whose launch fails")
    ap.add_argument("--plant", default="", help="planted speed multipliers, e.g. 2:1.10,4:0.9")
    ap.add_argument("--error-rate", type=float, default=0.0)
    ap.add_argument("--logs", default=str(HERE / "../m2-winners-20261001/runs/out-control"))
    args = ap.parse_args()
    if not args.mock:
        sys.exit("only --mock is runnable locally; Kaggle calls run_sweep() from the notebook")
    alpha = json.loads((HERE / "alpha.json").read_text())["alpha"]
    fail = {int(x) for x in args.fail_steps.split(",") if x}
    budget = args.budget_s if args.budget_s is not None else 6 * 3600
    verdict = run_sweep(
        factory=MockFactory(args.port, args.time_scale, fail,
                            {int(k): float(v) for k, v in (x.split(":") for x in args.plant.split(",") if x)},
                            args.error_rate),
        base_url=f"http://127.0.0.1:{args.port}", out=args.out,
        workload=HERE / "workload.json", logs_dir=args.logs, alpha=alpha, model="mock",
        deadline_monotonic=time.monotonic() + budget / args.time_scale,
        time_scale=args.time_scale, provenance={"source": "local mock rehearsal"})
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()
