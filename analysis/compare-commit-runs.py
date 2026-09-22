"""Compare two commit runs game-by-game: is the challenger smarter, and does the clock bind?

usage: compare-commit-runs.py <challenger-kaggle-output> <reference-kaggle-output>

Built for 3.8-vs-3.6 at the 7200 s submission cap. The 600 s probe arms could not separate
capability from speed because every episode was clock-limited. At 7200 s the question becomes
answerable, and this reports both halves:

  capability  -- levels per game, games with >=1 level, actions to first level
  clock       -- the minute each level landed, and whether games were still progressing at the cap

If the challenger's levels land later but still land, its slower turn is affordable here.
If they land later and stop landing, the clock binds and speed is the real constraint.
"""
from __future__ import annotations

import glob
import json
import re
import statistics as st
import sys

STEP = re.compile(r"^--- analysis_step=(\d+) \| action=(\d+) \| (\d\d):(\d\d):(\d\d)", re.M)


def episodes(out: str) -> dict[str, dict]:
    rows = {}
    for ev_path in sorted(glob.glob(f"{out}/artifacts/*_events.jsonl")):
        stem = ev_path.split("/")[-1].replace("_events.jsonl", "")
        game = stem.rsplit("_", 1)[0].split("-")[0]
        ev = [json.loads(l) for l in open(ev_path)]
        acts = [e for e in ev if e.get("type") == "action"]
        levels = max((int(e.get("level") or 1) for e in ev), default=1) - 1
        try:
            text = open(f"{out}/transcripts/{stem}.txt").read()
        except FileNotFoundError:
            text = ""
        steps = STEP.findall(text)
        idx = [int(s[0]) for s in steps]
        secs = [int(h) * 3600 + int(m) * 60 + int(s) for _, _, h, m, s in steps]
        t0, span = (secs[0], secs[-1] - secs[0]) if secs else (0, 0)
        mins, first_actions = [], None
        for i, e in enumerate(acts):
            if not e.get("level_completed"):
                continue
            if first_actions is None:
                first_actions = i + 1
            stp = e.get("analysis_step")
            if stp in idx:
                mins.append(round((secs[idx.index(stp)] - t0) / 60))
        rows[game] = dict(levels=levels, actions=len(acts), turns=len(steps), span_min=span / 60,
                          s_per_turn=(span / len(steps)) if steps else 0.0,
                          level_minutes=mins, actions_to_l1=first_actions,
                          changed=sum(1 for e in acts if e.get("board_changed")))
    return rows


def main(chal: str, ref: str) -> None:
    A, B = episodes(chal), episodes(ref)
    games = sorted(set(A) | set(B))
    print(f"{'game':<7}{'chal lv':>8}{'ref lv':>7}   {'chal level minutes':<24}{'ref level minutes'}")
    for g in games:
        a, b = A.get(g), B.get(g)
        am = str(a["level_minutes"]) if a else "-"
        bm = str(b["level_minutes"]) if b else "-"
        print(f"{g:<7}{(a['levels'] if a else 0):>8}{(b['levels'] if b else 0):>7}   {am:<24}{bm}")

    def agg(R, name):
        lv = sum(r["levels"] for r in R.values())
        got = sum(1 for r in R.values() if r["levels"] >= 1)
        spt = [r["s_per_turn"] for r in R.values() if r["s_per_turn"]]
        a2l = [r["actions_to_l1"] for r in R.values() if r["actions_to_l1"]]
        mins = [m for r in R.values() for m in r["level_minutes"]]
        acts = sum(r["actions"] for r in R.values())
        chg = sum(r["changed"] for r in R.values())
        print(f"\n{name}: {lv} levels over {got}/{len(R)} games | {acts} actions, {100*chg/max(acts,1):.0f}% changed board")
        if spt:
            print(f"  median s/turn {st.median(spt):.1f} | median span {st.median([r['span_min'] for r in R.values()]):.0f} min")
        if a2l:
            print(f"  actions to first level: median {st.median(a2l):.0f} (n={len(a2l)})")
        if mins:
            print(f"  level minutes: median {st.median(mins):.0f}, last {max(mins)}; "
                  f"{sum(1 for m in mins if m>78)} landed after minute 78")
        return lv

    la = agg(A, "challenger")
    lb = agg(B, "reference ")
    print(f"\nlevels {la} vs {lb}  ->  delta {la-lb:+d}")
    print("Clock read: if the challenger's level minutes run later but levels still land, its")
    print("slower turn is affordable at this cap; if they thin out past the reference's last")
    print("level minute, the clock is binding and speed is the constraint.")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
