#!/usr/bin/env python3
"""Score q5. Digest + S1 (reach_scorer) + S2 (recorded)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import reach_scorer as R

DIGEST = "7e0c1f1cdcd0d594b1bd21dee0fd74c9624a5a66141575d2d9f947fc298b433f"
TERM = frozenset({"level_up", "terminal"})


def main() -> int:
    run = Path(sys.argv[1]).resolve()
    seed = json.loads((run / "ipc/seeded_helpers.json").read_text())
    got = seed.get("symmetry.py")
    if got != DIGEST:
        print("INVALID: digest", got)
        return 2
    print(f"digest ok {got[:12]}…")

    use = re.compile(r"\bsymmetry\s*\.\s*(\w+)")
    cs, ex = R.cells(run), R.executions(run)
    executed_steps = set()
    names = set()
    n_reach = n_ok = 0
    for step, rnd, k, src in cs:
        hits = use.findall(src)
        if not hits:
            continue
        n_reach += 1
        ok, how = R.resolve(ex.get(step), src)
        print(f"  [{'ok ' if ok else 'ERR'}] step {step} round {rnd} ({how}): {sorted(set(hits))}")
        if ok:
            n_ok += 1
            executed_steps.add(step)
            names.update(hits)
    print(f"S1 reached={n_reach} executed={n_ok} functions={sorted(names) or '-'}")

    batches = []
    p = run / "ipc/actions.jsonl"
    if p.exists():
        batches = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    s2 = 0
    for b in batches:
        acts = b.get("actions") or []
        path = b.get("predicted_path") or []
        if not (1 <= len(acts) <= 5):
            continue
        if not any(isinstance(e, dict) and e.get("expect") in TERM for e in path):
            continue
        # env step of this submit is not in the batch; use turn as a coarse join
        # via agent_turn events if present.
        s2 += 1
        print(f"  disc batch turn={b.get('turn')} n={len(acts)}")
    # tighter S2: disc batch whose turn's env step had an executed reach
    turns = {}
    for f in run.glob("agent_logs/**/events.jsonl"):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            o = json.loads(line)
            if o.get("event") == "agent_turn":
                turns[o.get("turn")] = o.get("step")
    s2_joined = 0
    for b in batches:
        acts = b.get("actions") or []
        path = b.get("predicted_path") or []
        if not (1 <= len(acts) <= 5):
            continue
        if not any(isinstance(e, dict) and e.get("expect") in TERM for e in path):
            continue
        st = turns.get(b.get("turn"))
        if st in executed_steps:
            s2_joined += 1
    print(f"S2 executed-reach + cheap terminal-class batch: {s2_joined}")
    ow = 0
    rp = run / "ipc/rejections.jsonl"
    if rp.exists():
        ow = sum(
            1
            for l in rp.read_text().splitlines()
            if l.strip() and json.loads(l).get("reason") == "seeded_helper_overwrite"
        )
    print(f"overwrite rejections: {ow}")
    print("VERDICT:", "S1 PASS" if n_ok >= 1 else "S1 FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
