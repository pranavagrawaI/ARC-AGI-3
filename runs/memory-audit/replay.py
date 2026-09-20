"""Offline memory-path audit over Duck's authoritative per-game analyzer transcripts.

Replays the deployed note store (tool_agent._extract_scientist_note + the level/game-over clearing
policy) attempt by attempt and compares the predicted "note available" state with what the next
USER PROMPT actually carried ("Working world model carried from earlier turns").

usage: replay.py <kaggle-output dir>
Caveat: clears are detected from tool-result text; batched multi-action results that end a level
print differently and are missed (shows up as predicted=True/observed=False right after a level).
"""
import collections, glob, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ARC3-Inference"))
from inference.agent.tool_agent import _extract_scientist_note  # noqa: E402

KEYS = ("world_model", "goal_model", "action_model", "recent_findings", "open_questions", "current_plan")


def replay(path):
    txt = open(path, encoding="utf-8").read()
    attempts = re.split(r"(?m)^--- analysis_step=(\d+) \| action=\d+ \| [0-9:]+ \| tool-agent ---$", txt)
    store = {k: "" for k in KEYS}; rows = []
    for i in range(1, len(attempts) - 1, 2):
        step, body = int(attempts[i]), attempts[i + 1]
        parts = re.split(r"(?m)^\[([A-Z :a-z]+)\]$", body)
        secs = [(parts[j], parts[j + 1]) for j in range(1, len(parts) - 1, 2)]
        first_user = next((c for l, c in secs if l == "USER PROMPT"), "")
        observed = "Working world model carried from earlier turns" in first_user
        predicted = any(store.values())
        emitted = False; cleared = ""
        for l, c in secs:
            if l == "ASSISTANT":
                note = _extract_scientist_note(c)
                if any(note.get(k) for k in KEYS):
                    emitted = True
                    for k in KEYS:
                        if note.get(k):
                            store[k] = note[k]
            elif l.startswith("TOOL RESULT"):
                if re.search(r"(?m)^level_completed: true", c) or "progressed to a new level" in c:
                    cleared = "level"
                elif (re.search(r"(?m)^game_over: true", c) or "The game is over." in c) and not cleared:
                    cleared = "game_over"
        lost = bool(cleared) and any(store.values())
        if cleared:
            store = {k: "" for k in KEYS}
        rows.append(dict(step=step, observed=observed, predicted=predicted, emitted=emitted, cleared=cleared, lost=lost))
    return rows


def main(out):
    tot = collections.Counter()
    print(f"{'episode':18} {'attempts':>8} {'notes':>5} {'in-prompt':>9} {'mismatch':>8} {'lost@clear':>10}  emission pattern")
    for path in sorted(glob.glob(os.path.join(out, "transcripts", "*.txt"))):
        rows = replay(path); n = len(rows)
        em = sum(r["emitted"] for r in rows); inp = sum(r["observed"] for r in rows)
        mm = sum(r["observed"] != r["predicted"] for r in rows); lost = sum(r["lost"] for r in rows)
        pat = []
        for r in rows:
            ch = "E" if r["emitted"] else "."
            if pat and pat[-1][0] == ch: pat[-1][1] += 1
            else: pat.append([ch, 1])
        tot.update(dict(attempts=n, notes=em, inprompt=inp, mismatch=mm, lost=lost))
        tot["tool-only episodes (<=1 note)"] += em <= 1
        print(f"{os.path.basename(path)[:18]:18} {n:8} {em:5} {inp:9} {mm:8} {lost:10}  {' '.join(f'{c}{k}' for c, k in pat)[:60]}")
    print(dict(tot))


if __name__ == "__main__":
    main(sys.argv[1])
