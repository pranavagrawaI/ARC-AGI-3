"""Readback for the Step 2 verification kernel: firings by cause, cost, uptake, stall coverage."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

out = Path(sys.argv[1] if len(sys.argv) > 1 else "kaggle-output")
traces = sorted(out.glob("transcripts/*_critic.jsonl"))
events = {p.name.split("_events")[0]: p for p in out.glob("artifacts/*_events.jsonl")}
print(f"critic traces: {len(traces)}; event files: {len(events)}")

for trace in traces:
    stem = trace.name.replace("_critic.jsonl", "")
    firings = [json.loads(l) for l in trace.read_text().splitlines() if l.strip()]
    ev = [json.loads(l) for l in events[stem].read_text().splitlines()] if stem in events else []
    actions = [e for e in ev if e.get("type") == "action"]
    levels = max((int(e.get("level") or 1) for e in ev), default=1) - 1
    # longest run of no-change executed actions, from events
    run = best = 0
    for e in actions:
        run = run + 1 if not e.get("board_changed") else 0
        best = max(best, run)
    print(f"\n== {stem}: {len(actions)} actions, levels={levels}, longest no-change run={best}, firings={len(firings)}")
    print("   causes:", dict(Counter(f["cause"] for f in firings)))
    for f in firings:
        q = (f.get("report_text") or "").split("\n")
        q = next((l for l in q if l.startswith("question:")), "")[:110]
        print(f"   a{f['action_num']:>3} L{f['level']} {f['cause']:<18} {f['seconds']:>6.1f}s tok={f['prompt_tokens']}+{f['completion_tokens']} parsed={f['parsed']} fb={f['fallback']} rec={f.get('recommended_actions')} legal={f.get('recommended_legal')} err={(f.get('error') or '')[:60]}")
        print(f"        {q}")
    secs = sum(f["seconds"] for f in firings)
    print(f"   critic wall: {secs:.1f}s total")
    if best >= 3 and not any(f["cause"] == "stall_no_change" for f in firings):
        print("   !! no-change run >=3 in events but no stall_no_change firing")
    if len(actions) >= 25 and not any(f["cause"] == "stall_no_progress" for f in firings) and levels == 0:
        print("   !! >=25 actions without progress but no stall_no_progress firing")
