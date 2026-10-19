"""What does the model do right after B1's contradictory animated no-op turn?

A turn is "contradictory" when its user message says the last action "animated ... but it
was not a no-op" AND carries a no-op guard stop for that same action. The comparison group
is turns with a no-op guard stop and no animation line. For each turn we read all the model's
python code until the next user message and count:
  - repeat: any action() call in the turn starts with the stopped action
  - reads_anim: the code touches last_animation_timeline / last_animation_frames
  - blocked: the following tool output reports a known-no-op BLOCK or StaleStateActionError
Usage: python3 -I contradiction_followups.py OUTPUT_DIR [OUTPUT_DIR ...]
"""
import collections
import glob
import json
import re
import sys

ANIM = "but it was not a no-op"
STOP = re.compile(r"stopped by the no-op guard: action '([^']+)'")
ACTION_CALL = re.compile(r"action\(\s*\[\s*['\"]?([A-Z0-9_]+)")


def text(message):
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(x.get("text", "") for x in content if isinstance(x, dict))
    return ""


def code_of(message):
    out = []
    for call in message.get("tool_calls") or []:
        args = call.get("function", {}).get("arguments", "")
        try:
            out.append(json.loads(args).get("code", args))
        except (ValueError, AttributeError):
            out.append(args)
    return "\n".join(out)


def followups(path):
    requests = []
    with open(path) as fh:
        for line in fh:
            record = json.loads(line)
            if record.get("event") == "request":
                requests.append(record["messages"])
    for i, messages in enumerate(requests[:-1]):
        user = text(messages[-1])
        stop = STOP.search(user)
        if not stop:
            continue
        group = "animated" if ANIM in user else "plain"
        # the whole turn: every model call up to (not including) the next user message
        later = None
        for nxt in requests[i + 1:]:
            if len(nxt) <= len(messages) or text(nxt[len(messages) - 1]) != user:
                break  # transcript was compacted; can't align past here
            later = nxt
            if nxt[-1].get("role") == "user":
                break
        if later is None:
            continue
        tail = [m for m in later[len(messages):] if m.get("role") != "user"]
        code = "\n".join(code_of(m) for m in tail if m.get("role") == "assistant")
        tool_out = "\n".join(text(m) for m in tail if m.get("role") == "tool")
        first = ACTION_CALL.search(code)
        calls = [m.group(1) for m in ACTION_CALL.finditer(code)]
        yield group, {
            "repeat": stop.group(1).split()[0] in calls,
            "reads_anim": "last_animation_" in code,
            "blocked": "BLOCKED (known no-op" in tool_out or "StaleStateActionError" in tool_out,
            "acted": bool(first),
        }


def main(dirs):
    totals = collections.defaultdict(collections.Counter)
    for d in dirs:
        for path in sorted(glob.glob(f"{d}/*_requests.jsonl")):
            for group, flags in followups(path):
                totals[group]["n"] += 1
                for key, value in flags.items():
                    totals[group][key] += value
    result = {g: dict(c) for g, c in totals.items()}
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
