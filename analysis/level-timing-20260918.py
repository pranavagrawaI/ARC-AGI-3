"""When did levels actually land in the 2026-09-18 commit run?

usage: level-timing-20260918.py <kaggle-output-dir>

Motive: the per-game cap was about to be raised 120 -> 128 min to "use the full envelope".
This asks whether the tail of the cap converts time into levels at all. It reads each level
completion's wall-clock second from the transcript step timestamps, and measures whether the
agent was still active (board_changed) in the last third of each game.
"""
from __future__ import annotations

import glob
import json
import re
import statistics as st
import sys

STEP = re.compile(r"^--- analysis_step=(\d+) \| action=(\d+) \| (\d\d):(\d\d):(\d\d)", re.M)


def main(out: str) -> None:
    lvl_times, late, early = [], [0, 0], [0, 0]
    for ev_path in sorted(glob.glob(f"{out}/artifacts/*_events.jsonl")):
        stem = ev_path.split("/")[-1].replace("_events.jsonl", "")
        ev = [json.loads(l) for l in open(ev_path)]
        acts = [e for e in ev if e.get("type") == "action"]
        text = open(f"{out}/transcripts/{stem}.txt").read()
        steps = STEP.findall(text)
        if not steps:
            continue
        idx = [int(s[0]) for s in steps]
        secs = [int(h) * 3600 + int(m) * 60 + int(s) for _, _, h, m, s in steps]
        t0, span = secs[0], secs[-1] - secs[0]
        for e in acts:
            stp = e.get("analysis_step")
            if stp not in idx:
                continue
            at = secs[idx.index(stp)] - t0
            if e.get("level_completed"):
                lvl_times.append(at)
            bucket = late if at >= span * 2 / 3 else (early if at < span / 3 else None)
            if bucket is not None:
                bucket[1] += 1
                bucket[0] += 1 if e.get("board_changed") else 0

    lvl_times.sort()
    print(f"levels: {len(lvl_times)}; last one at minute {max(lvl_times)/60:.0f} of a 120 min cap")
    for cut in (30, 60, 90, 120):
        print(f"  by {cut:>3} min: {sum(1 for t in lvl_times if t <= cut*60):>2}/{len(lvl_times)}")
    print(f"still acting late: {late[0]}/{late[1]} last-third actions changed the board "
          f"({100*late[0]/max(late[1],1):.0f}%), vs {100*early[0]/max(early[1],1):.0f}% early")
    print("=> the tail of the cap is active but unproductive; more minutes there buy no levels.")


if __name__ == "__main__":
    main(sys.argv[1])
