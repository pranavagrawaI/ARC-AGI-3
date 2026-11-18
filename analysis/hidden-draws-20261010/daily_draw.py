"""Standing daily draw: submit today's queued candidate, else the fallback (control again since 10-25).
Exact-version submission quarantine takes precedence and can leave the slot unused.

  daily_draw.py            (run by arc-hidden-draw-daily.timer, hourly at :30 UTC)

queue.json maps a UTC date to {"kernel", "version", "message"[, "requires": ["D11", ...]]}; a candidate goes there
only after its smoke passed. With "requires", it is submitted only once those draws are COMPLETE with a score above 0.
Draw labels are D4 on 2026-10-13, D5 on 2026-10-14, ... so every day has one stable tag and submit_draw.py
stays idempotent. Plan (2026-10-13, Pranav): gold target; the slot goes to the candidate that resolves a path
fastest; second draws only on slack days and only for arms whose first draw was >= 30; from 2026-10-30
draws confirm the two finals (anchor = incumbent, wildcard = best mean draw).
"""
import datetime as dt, json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KAGGLE_PY = "/home/pranav/.local/share/uv/tools/kaggle/bin/python"
REQUIRES_FAILED, REQUIRES_PENDING = 3, 4  # submit_draw.py exit codes for a queue entry's "requires"
FALLBACK_HOUR = 10  # UTC (15:30 IST); was 20, but 20-24Z is 01:30-05:30 IST when the laptop sleeps: D6 (10-15) was lost
# From 2026-10-19 the empty-queue fallback is the newest no-harm candidate, not the control (Pranav, 10-18):
# five control draws anchor the finals already, a sixth adds no information, and an equal-ticket candidate does.
# The private LB is the other 55 games of the same run (draw-variance-20261018), so every draw is a finals ticket.
# F1x is also the incumbent stack in incumbent/stack.json since 2026-10-20.
# From 2026-10-25 the fallback is the control again: F1x v1 is quarantined, so its fallback left D13-D15 unused.
# An unused slot is a lost finals ticket; the control is 0/5 zeros (27.8 +/- 3.2).
FALLBACK = ("pranavagrawai/dfranzen-m2-raw", "1",
            "(fallback control, dfranzen M2 raw, no new candidate queued): unchanged control, "
            "5 prior hidden draws 27.8 +/- 3.2, no zeros.")


def main():
    today = dt.datetime.now(dt.timezone.utc).date()
    if today > dt.date(2026, 11, 2):
        return  # competition over
    draw = f"D{4 + (today - dt.date(2026, 10, 13)).days}"
    queue = json.loads((HERE / "queue.json").read_text()) if (HERE / "queue.json").exists() else {}
    entry = queue.get(today.isoformat())
    yesterday = f"D{3 + (today - dt.date(2026, 10, 13)).days}"
    log_text = (HERE / "draws.log").read_text() if (HERE / "draws.log").exists() else ""
    if f" {draw}: " not in log_text and f" {yesterday}: submitted" not in log_text and f" {yesterday}: skip, already" not in log_text:
        with open(HERE / "draws.log", "a") as f:  # once per day: the first run that finds yesterday's slot unused
            f.write(f"{dt.datetime.now(dt.timezone.utc).isoformat()} {draw}: note, {yesterday} slot went unused\n")
    if not entry and dt.datetime.now(dt.timezone.utc).hour < FALLBACK_HOUR:
        return  # leave the day open for a candidate that passes its smoke later

    def submit(kernel, version, message, requires=()):
        cmd = [KAGGLE_PY, "-B", str(HERE / "submit_draw.py"), draw, kernel, version, message, "--not-before", today.isoformat()]
        return subprocess.run(cmd + (["--requires", ",".join(requires)] if requires else []), cwd=HERE).returncode

    if not entry:
        sys.exit(submit(*FALLBACK))
    rc = submit(entry["kernel"], str(entry["version"]), entry["message"], entry.get("requires", ()))
    if fall_back(rc, dt.datetime.now(dt.timezone.utc).hour):
        why = "failed" if rc == REQUIRES_FAILED else "still pending at the fallback hour"
        kernel, version, message = FALLBACK
        rc = submit(kernel, version, message.replace("no new candidate queued",
                                                     f"queued {entry['kernel']} held: {','.join(entry['requires'])} {why}"))
    sys.exit(0 if rc == REQUIRES_PENDING else rc)


def fall_back(rc, hour):
    """A queued candidate whose required draws failed yields the day at once; one still pending waits until the
    fallback hour, so a slow prerequisite can't cost the day (the laptop sleeps 17:30Z-03:00Z)."""
    return rc == REQUIRES_FAILED or (rc == REQUIRES_PENDING and hour >= FALLBACK_HOUR)


if __name__ == "__main__":
    main()
