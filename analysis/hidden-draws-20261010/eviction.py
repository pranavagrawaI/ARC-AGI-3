"""Was the decisive evidence still in context? Offline, from request logs (no GPU).

  python3 -B eviction.py

The decisive action for (game, level) is a sibling run's level-completing action, matched as in
metric-validation/stuck.py (same id; clicks within 2 cells). For every level attempt in a run that has
request logs, record:
  entry_evicted     the level's first step was no longer in the prompt at the attempt's last request
  contact_step      the first step at which the attempt took a decisive action (stuck attempts only)
  contact_evicted   that step was no longer in the prompt at the attempt's last request
"Visible" means the step's "Current state: step N" line is still present somewhere in the request messages.
"""
import glob, json, re, statistics
from pathlib import Path
from redraw_bound import ROOT, RUNS

LOGGED = ["c1", "pc", "pt"]
STEP = re.compile(r"Current state: step (\d+), level (\d+)\.")


def text_of(content):
    if isinstance(content, str):
        return content
    return "\n".join(p.get("text", "") for p in content or [] if isinstance(p, dict))


def visibility(path):
    """List of (decision step, min visible step) per request, in order."""
    out = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            steps = [int(m.group(1)) for msg in r["messages"] for m in STEP.finditer(text_of(msg.get("content")))]
            if steps:
                out.append((max(steps), min(steps)))
    return out


def segments(run):
    """(game, level) -> (solved, [(action id, data, global step)]) with step = history index + 1."""
    b = json.loads((ROOT / RUNS[run] / "benchmark.json").read_text())
    out = {}
    for r in b["game_runs"]:
        h, apl, done = r["history"], r["actions_per_level"], int(r["levels_completed"] or 0)
        k = 0
        for L, n in enumerate(apl):
            if L > done:
                break
            seg = h[k:k + n] if L < done else h[k:]
            out[(r["game_id"][:4], L)] = (L < done, [(x["action"]["id"], x["action"].get("data") or {}, k + i + 1) for i, x in enumerate(seg)], k + 1)
            k += n
    return out


def same(a, b):
    if a[0] != b[0]:
        return False
    if a[0] != "ACTION6":
        return True
    return abs(a[1].get("x", -99) - b[1].get("x", 99)) <= 2 and abs(a[1].get("y", -99) - b[1].get("y", 99)) <= 2


def main():
    S = {r: segments(r) for r in RUNS}
    stuck_rows, solved_rows = [], []
    for run in LOGGED:
        for f in glob.glob(str(ROOT / RUNS[run] / "*_requests.jsonl")):
            game = Path(f).name[:4]
            vis = visibility(f)
            for (g, L), (solved, seq, entry) in S[run].items():
                if g != game or not seq:
                    continue
                last = seq[-1][2]
                at_end = [v for d, v in vis if entry <= d <= last + 1]
                if not at_end:
                    continue
                v_end = at_end[-1]
                row = dict(game=g, L=L, n=len(seq), entry_evicted=v_end > entry)
                if solved:
                    solved_rows.append(row); continue
                decisive = [S[q][(g, L)][1][-1] for q in RUNS if q != run and (g, L) in S[q] and S[q][(g, L)][0] and S[q][(g, L)][1]]
                if not decisive:
                    continue
                contact = next((s for a, d, s in seq if any(same((a, d), (x[0], x[1])) for x in decisive)), None)
                row.update(contact_step=contact, contact_evicted=(contact is not None and v_end > contact))
                stuck_rows.append(row)
    long_solved = [r for r in solved_rows if r["n"] >= 30]
    print(f"solved levels: {len(solved_rows)}; entry evicted before the solve: {sum(r['entry_evicted'] for r in solved_rows)}"
          f"  (of {len(long_solved)} with >=30 actions: {sum(r['entry_evicted'] for r in long_solved)})")
    print(f"stuck levels with a sibling solve: {len(stuck_rows)}; entry evicted by the end: {sum(r['entry_evicted'] for r in stuck_rows)}")
    c = [r for r in stuck_rows if r["contact_step"] is not None]
    print(f"  took a decisive action: {len(c)}; that contact evicted by the end: {sum(r['contact_evicted'] for r in c)}")
    for r in stuck_rows:
        print("   ", r)


if __name__ == "__main__":
    main()
