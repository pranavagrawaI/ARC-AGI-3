"""Server-ready and precache read times vs run date, from every retained Kaggle log."""
import re, sys, json, hashlib
from pathlib import Path
rows, seen = [], set()
for f in Path(sys.argv[1]).read_text().split():
    p = Path(f); b = p.read_bytes()
    h = hashlib.sha1(b).hexdigest()
    if h in seen: continue
    seen.add(h)
    t = b.decode(errors="replace").replace("\\n", "\n")
    ready = [int(x) for x in re.findall(r"READY after (\d+)s", t)]
    pre = re.findall(r"Precaching: loaded (\d+) files \(([\d.]+) GB\) in ([\d.]+) seconds", t)
    dates = re.findall(r"(20\d\d-\d\d-\d\d)[ T](\d\d:\d\d)", t)
    gpu = re.search(r"(RTX PRO 6000|Tesla T4|H100|L4)", t)
    rows.append({"file": f, "date": min(d[0] + " " + d[1] for d in dates) if dates else None,
                 "ready_s": ready, "precache": [(float(g), float(s)) for _, g, s in pre],
                 "gpu": gpu.group(1) if gpu else None})
rows.sort(key=lambda r: r["date"] or "")
json.dump(rows, open(sys.argv[2], "w"), indent=1)
for r in rows:
    pc = " ".join(f"{g:.0f}GB/{s:.0f}s" for g, s in r["precache"])
    print(r["date"], r["ready_s"], pc, r["gpu"], r["file"][-70:])
