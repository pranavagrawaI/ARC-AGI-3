"""periodic-v1 screen readback: primary metric, decision rule, and the delivery funnel.

usage: readback.py control/kaggle-output candidate/kaggle-output [--cap-seconds 3600] [--delta 8]

Funnel per critic attempt (derived from events.jsonl + the per-game critic trace, no extra
game-loop state): attempted -> report generated (parsed | fallback) or escaped exception ->
report included in a submitted actor request ([CRITIC REPORT] with the packet id in that
analysis step's transcript) -> actor response completed ([ANALYZER STATUS] written for that step) ->
a subsequent environment action recorded for that step.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_events(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def episode(out: Path, ev_path: Path) -> dict:
    stem = ev_path.name.replace("_events.jsonl", "")
    ev = load_events(ev_path)
    acts = [e for e in ev if e.get("type") == "action"]
    levels = max((int(e.get("level") or 1) for e in ev), default=1) - 1
    steps_with_action = {e.get("analysis_step") for e in acts}
    analyses = defaultdict(list)
    for e in ev:
        if e.get("type") == "analysis":
            analyses[e.get("analysis_step")].append(e.get("transcript") or "")
    trace = out / "transcripts" / f"{stem}_critic.jsonl"
    firings = load_events(trace) if trace.exists() else []
    critic_events = [e for e in ev if e.get("type") == "critic"]
    escaped = [e for e in critic_events if e.get("error")]
    funnel = Counter(attempted=len(critic_events), escaped=len(escaped))
    exposures = []
    for e in critic_events:
        if e.get("error"):
            continue
        step = e.get("analysis_step")
        pid = e.get("packet_id") or ""
        texts = analyses.get(step, [])
        included = any("[CRITIC REPORT]" in t and pid in t for t in texts)
        completed = any("[ANALYZER STATUS]" in t for t in texts)  # the actor turn returned (any status)
        acted = step in steps_with_action
        f = next((x for x in firings if x.get("packet_id") == pid), {})
        funnel["generated"] += 1
        funnel["fallback"] += bool(f.get("fallback"))
        funnel["included"] += included
        funnel["actor_completed"] += completed
        funnel["acted_after"] += acted
        exposures.append(dict(step=step, action_num=e.get("action_num"), included=included, completed=completed, acted=acted, seconds=f.get("seconds")))
    uptake = 0
    for f in firings:
        rec = (f.get("recommended_actions") or [None])[0]
        n = f["action_num"]
        if rec and n < len(acts):
            disp = acts[n].get("action_display") or ""
            if rec.get("action") == "MOUSE":
                uptake += disp == f"MOUSE(row={rec.get('row')}, col={rec.get('col')})"
            else:
                uptake += disp == rec.get("action")
    return dict(
        stem=stem, game=stem.split("_")[0].split("-")[0], levels=levels, actions=len(acts),
        turns=len(steps_with_action), attempts=len(critic_events), delivered=funnel["included"],
        funnel=dict(funnel), exposures=exposures, uptake=uptake,
        critic_s=round(sum(float(f.get("seconds") or 0) for f in firings), 1),
        causes=dict(Counter(f.get("cause") for f in firings)),
    )


def arm(out: Path) -> dict[str, dict]:
    return {r["stem"]: r for r in (episode(out, p) for p in sorted(out.glob("artifacts/*_events.jsonl")))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("control"); ap.add_argument("candidate")
    ap.add_argument("--cap-seconds", type=float, default=3600.0)
    ap.add_argument("--delta", type=int, default=8)
    a = ap.parse_args()
    arms = {"control": arm(Path(a.control)), "candidate": arm(Path(a.candidate))}

    print(f"{'episode':<22}{'control':>16}{'candidate':>30}")
    for s in sorted({s for r in arms.values() for s in r}):
        c, k = arms["control"].get(s), arms["candidate"].get(s)
        left = "(missing)" if c is None else f"L{c['levels']} {c['actions']:>3}a {c['turns']:>2}t"
        right = "(missing)" if k is None else f"L{k['levels']} {k['actions']:>3}a {k['turns']:>2}t  att{k['attempts']} dlv{k['delivered']} up{k['uptake']}"
        print(f"{s:<22}{left:>16}{right:>30}")

    totals, by_game = {}, {}
    for name, rows in arms.items():
        totals[name] = sum(r["levels"] for r in rows.values())
        g = defaultdict(int)
        for r in rows.values():
            g[r["game"]] += r["levels"]
        by_game[name] = dict(g)
        print(f"\n{name}: episodes={len(rows)} TOTAL LEVELS={totals[name]} episodes>=1 level={sum(r['levels'] > 0 for r in rows.values())} "
              f"by game={by_game[name]} actions={sum(r['actions'] for r in rows.values())} mean turns={sum(r['turns'] for r in rows.values())/max(1, len(rows)):.1f}")

    rows = arms["candidate"]
    funnel = Counter()
    for r in rows.values():
        funnel.update(r["funnel"])
    cs = sum(r["critic_s"] for r in rows.values())
    print(f"candidate funnel: {dict(funnel)}  critic wall={cs:.0f}s ({100 * cs / max(1, a.cap_seconds * len(rows)):.1f}% of cap) "
          f"uptake={sum(r['uptake'] for r in rows.values())} causes={dict(sum((Counter(r['causes']) for r in rows.values()), Counter()))}")
    dist = Counter(r["delivered"] for r in rows.values())
    print(f"delivered reviews per episode: {dict(sorted(dist.items()))}  (episodes with >=2 delivered: {sum(v for k, v in dist.items() if k >= 2)}/{len(rows)})")

    delta = totals["candidate"] - totals["control"]
    dg = {g: by_game["candidate"].get(g, 0) - by_game["control"].get(g, 0) for g in set(by_game["control"]) | set(by_game["candidate"])}
    best = max(dg.values(), default=0)
    rule = delta >= a.delta and (delta - max(best, 0)) > 0
    print(f"\nDELTA = {totals['candidate']} - {totals['control']} = {delta:+d}; per game {dg}; minus best game {delta - max(best, 0):+d}")
    print(f"advance periodic-v1: {'YES' if rule else 'NO'}  (rule: delta >= {a.delta} and delta - max positive game contribution > 0)")


if __name__ == "__main__":
    main()
