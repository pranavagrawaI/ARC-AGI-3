"""Standing daily draw: submit today's queued candidate, else the incumbent. Something runs every day.

  daily_draw.py            (run by arc-hidden-draw-daily.timer, hourly at :30 UTC)

queue.json maps a UTC date to {"kernel", "version", "message"}; a candidate goes there only after its smoke passed.
Draw labels are D4 on 2026-10-13, D5 on 2026-10-14, ... so every day has one stable tag and submit_draw.py
stays idempotent. Plan (2026-10-13, Pranav): gold target; the slot goes to the candidate that resolves a path
fastest; second draws only on slack days and only for arms whose first draw was >= 30; from 2026-10-30
draws confirm the two finals (anchor = incumbent, wildcard = best mean draw).
"""
import datetime as dt, json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KAGGLE_PY = "/home/pranav/.local/share/uv/tools/kaggle/bin/python"
FALLBACK_HOUR = 10  # UTC (15:30 IST); was 20, but 20-24Z is 01:30-05:30 IST when the laptop sleeps: D6 (10-15) was lost
INCUMBENT = ("pranavagrawai/dfranzen-m2-raw", "1",
             "(incumbent, no candidate ready): unchanged dfranzen M2, byte-identical to 56747536/D2.")


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
    kernel, version, message = (entry["kernel"], str(entry["version"]), entry["message"]) if entry else INCUMBENT
    sys.exit(subprocess.run([KAGGLE_PY, "-B", str(HERE / "submit_draw.py"), draw, kernel, version, message,
                             "--not-before", today.isoformat()], cwd=HERE).returncode)


if __name__ == "__main__":
    main()
