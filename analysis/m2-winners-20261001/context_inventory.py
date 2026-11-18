"""What fills the context window? Composition of the request context by message part, late in each game."""
import json, sys, collections, re
from pathlib import Path

def parts(msg):
    role = msg["role"]; c = msg.get("content")
    out = collections.Counter()
    if isinstance(c, str):
        out[role + ":text"] += len(c)
    elif isinstance(c, list):
        for p in c:
            if p.get("type") == "text": out[role + ":text"] += len(p.get("text", ""))
            else: out[role + ":image"] += 1
    for k in ("reasoning_content", "reasoning"):
        if msg.get(k): out[role + ":thinking"] += len(msg[k])
    for tc in msg.get("tool_calls") or []:
        out[role + ":toolcall"] += len(tc["function"].get("arguments", ""))
    return out

def boiler(user_texts):
    """Lines that repeat verbatim across user messages = boilerplate."""
    cnt = collections.Counter(l for t in user_texts for l in set(t.splitlines()) if len(l) > 30)
    rep = {l for l, n in cnt.items() if n >= 3}
    tot = sum(len(t) for t in user_texts)
    b = sum(len(l) + 1 for t in user_texts for l in t.splitlines() if l in rep)
    return b, tot

if __name__ == "__main__":
    out = Path(sys.argv[1]); agg = collections.Counter(); B = T = 0
    for f in sorted(out.glob("*_p0_requests.jsonl")):
        reqs = [json.loads(l) for l in f.open()]
        reqs = [r for r in reqs if r["event"] == "request"]
        late = reqs[int(len(reqs) * 0.8)]          # a request 80% through the game
        c = collections.Counter()
        for m in late["messages"]: c += parts(m)
        ut = []
        for m in late["messages"]:
            if m["role"] == "user":
                cc = m["content"]; ut.append(cc if isinstance(cc, str) else " ".join(p.get("text", "") for p in cc if p.get("type") == "text"))
        b, t = boiler(ut); B += b; T += t
        tot = sum(v for k, v in c.items() if not k.endswith("image"))
        agg += c
        print(f"{f.name[:4]} msgs={len(late['messages']):3d} chars={tot:7,} " + " ".join(f"{k}={v/tot:.0%}" for k, v in sorted(c.items()) if not k.endswith('image')) + f" images={c['user:image']}  user-boilerplate={b/max(t,1):.0%} of user text")
    tot = sum(v for k, v in agg.items() if not k.endswith("image"))
    print("ALL  " + " ".join(f"{k}={v/tot:.0%}" for k, v in sorted(agg.items()) if not k.endswith('image')) + f"  user boilerplate={B/T:.0%}")
