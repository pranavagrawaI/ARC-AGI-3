"""Append/replace one hint in hints.json.  addhint.py <game-L<level>> <json with goal, plan, basis, plan_check>"""
import json, sys
from pathlib import Path
p = Path(__file__).with_name("hints.json")
h = json.loads(p.read_text()) if p.exists() else {}
key, entry = sys.argv[1], json.loads(sys.argv[2])
assert set(entry) >= {"goal", "basis"}, entry.keys()
h[key] = entry
p.write_text(json.dumps(h, indent=1, ensure_ascii=False) + "\n")
print(f"{len(h)} hints; wrote {key}")
