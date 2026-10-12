"""Startup timeline of every dfranzen-family kernel log we hold: disk precache, weight load, READY or DEADLINE."""
import json, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def text(path):
    raw = path.read_text(errors='replace')
    try:
        return ''.join(r.get('data', '') for r in json.loads(raw))
    except Exception:
        return raw


def row(log):
    out = text(log)
    ready = re.search(r'READY after (\d+)s', out)
    deadline = re.search(r'DEADLINE at (\d+)s', out)
    pre = [(int(n), float(gb), float(s)) for n, gb, s in re.findall(r'Precaching: loaded (\d+) files \(([\d.]+) GB\) in ([\d.]+) seconds', out)]
    big = max(pre, key=lambda p: p[1]) if pre else None
    serve = log.parent / 'serve.log'
    load = None
    if serve.exists():
        m = re.search(r'Load weight end\. elapsed=([\d.]+) s', serve.read_text(errors='replace'))
        load = float(m[1]) if m else None
    return {'log': str(log.relative_to(REPO)), 'ready_s': int(ready[1]) if ready else None,
            'deadline_s': int(deadline[1]) if deadline else None,
            'precache_gb': big[1] if big else None, 'precache_s': big[2] if big else None, 'weight_load_s': load}


if __name__ == '__main__':
    logs = [Path(p).resolve() for p in sys.argv[1:]]
    rows = [row(p) for p in logs]
    print(json.dumps(rows, indent=1))
