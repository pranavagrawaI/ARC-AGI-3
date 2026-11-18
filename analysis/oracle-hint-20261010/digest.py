"""Compact digest of one attempt's transcript, for classifying why it failed.

  python3 -B digest.py <attempt> [reason_chars]    e.g. digest.py tu93-L3-A1-r2 500

Prints the level's hint (goal, route length), then, in order, every distinct assistant message across the request
log: its reasoning (head ... tail), the code it ran (head), and the tool result (head). The request log holds requests
only, so the final response of the attempt is missing.
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def clip(s, n):
    s = str(s or "")
    return s if len(s) <= 2 * n else s[:n] + f" [...{len(s) - 2 * n} chars...] " + s[-n:]


def main(name, n=400):
    sess = "a0" if "-A0-" in name else "hints"
    key = name.rsplit("-A", 1)[0]
    h = json.loads((HERE / "hints.json").read_text()).get(key)
    if h:
        print(f"HINT goal: {h['goal']}\nROUTE: {len(h['route'])} actions ({h.get('route_source')})\n")
    seen, step = set(), None
    for line in (HERE / "sessions" / sess / "output" / f"{name}_p0_requests.jsonl").read_text().splitlines():
        req = json.loads(line)
        msgs = req["messages"]
        for i, m in enumerate(msgs):
            if m["role"] != "assistant":
                continue
            sig = (m.get("reasoning_content") or "")[:200] + str(m.get("tool_calls"))[:200] + str(m.get("content"))[:100]
            if sig in seen:
                continue
            seen.add(sig)
            if req["action"] != step:
                step = req["action"]
                print(f"\n######## turn {req['analysis_step']} action-count {req['action']}")
            print(f"-- R: {clip(m.get('reasoning_content'), n)}")
            if m.get("content"):
                print(f"-- C: {clip(m['content'], n // 2)}")
            for tc in (m.get("tool_calls") or []) if isinstance(m.get("tool_calls"), list) else [m.get("tool_calls")]:
                print(f"-- CODE: {clip(tc, n // 2)}")
            nxt = msgs[i + 1] if i + 1 < len(msgs) and msgs[i + 1]["role"] == "tool" else None
            if nxt:
                print(f"-- OUT: {clip(nxt['content'], n // 2)}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 400)
