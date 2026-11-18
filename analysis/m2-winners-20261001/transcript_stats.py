"""Turn-level waste counters from TAAF transcripts (<out>/transcripts/*.txt)."""
import re, sys, collections, statistics
from pathlib import Path

SEC = re.compile(r"^\[(SYSTEM PROMPT|USER PROMPT|MODEL RESPONSE META|THINKING|TOOL CALL: python|TOOL RESULT: python|ANALYZER STATUS)\]\s*$")

def blocks(text):
    cur, buf = None, []
    for line in text.splitlines():
        m = SEC.match(line)
        if m or line.startswith("--- analysis_step="):
            if cur: yield cur, "\n".join(buf)
            cur, buf = (m.group(1) if m else "TURN " + line), []
        else:
            buf.append(line)
    if cur: yield cur, "\n".join(buf)

def stats(path):
    s = collections.Counter(); think = []; errs = collections.Counter()
    for kind, body in blocks(Path(path).read_text(errors="replace")):
        if kind == "THINKING":
            think.append(len(body))
        elif kind == "TOOL RESULT: python":
            s["tool_results"] += 1
            if "Traceback" in body or body.lstrip().startswith("error"):
                s["tool_errors"] += 1
                m = re.findall(r"^(\w+Error|\w+Exception|TimeoutError)\b", body, re.M)
                errs[m[-1] if m else "other"] += 1
            if "timed out" in body.lower() or "timeout" in body.lower(): s["tool_timeouts"] += 1
            if "truncated" in body.lower(): s["truncated"] += 1
            if re.search(r"guard|blocked|refused", body, re.I): s["guard_or_refusal"] += 1
        elif kind == "ANALYZER STATUS":
            s["status"] += 1
            if "yield" in body.lower(): s["yields"] += 1
    return s, think, errs

if __name__ == "__main__":
    tot = collections.Counter(); allthink = []; allerr = collections.Counter()
    for p in sorted(Path(sys.argv[1]).glob("transcripts/*.txt")):
        s, t, e = stats(p); tot += s; allthink += t; allerr += e
        print(f"{p.name[:4]} results={s['tool_results']:3d} errors={s['tool_errors']:3d} timeouts={s['tool_timeouts']:2d} "
              f"guards={s['guard_or_refusal']:3d} yields={s['yields']:2d} think_chars(med)={statistics.median(t) if t else 0:7.0f} max={max(t) if t else 0}")
    print("TOTAL", dict(tot), " error share", f"{tot['tool_errors']/max(tot['tool_results'],1):.0%}")
    print("error types", allerr.most_common(10))
    q = sorted(allthink); n = len(q)
    print(f"thinking chars per response: n={n} p50={q[n//2]} p90={q[int(n*.9)]} p99={q[int(n*.99)]}; share of chars in top 10% = {sum(q[int(n*.9):])/sum(q):.0%}")
