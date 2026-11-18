"""Retrieve selected immutable wave-2 evidence; never touches launcher state."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent.parent / 'experiment'))
from collect_outputs import KaggleOutput

parser = argparse.ArgumentParser()
parser.add_argument('--arms', nargs='+', default=['lean-cap4', 'lean-prune'])
parser.add_argument('--games', nargs='*', default=[])
parser.add_argument('--names', nargs='+', default=['llm-trace.jsonl', 'tool-trace.jsonl', 'host-actions.jsonl', 'compute.json', 'context-projection.jsonl', 'worker-stderr.log', 'worker-metadata.json'])
parser.add_argument('--calls', nargs='*', type=int, default=[])
parser.add_argument('--workers', type=int, default=2)
args = parser.parse_args()
tasks = []
for arm in args.arms:
    index = json.loads((ROOT / arm / 'tycho-dfranzen/evidence-index.json').read_text())
    for row in index['files']:
        name = row['path']
        if args.games and not any('/'+g+'-' in name for g in args.games):
            continue
        if Path(name).name not in args.names:
            continue
        if args.calls and not any('/calls/'+f'{c:06d}/' in name for c in args.calls):
            continue
        tasks.append((arm, row))
local = threading.local()
def fetch(task):
    arm, row = task
    path = ROOT / arm / row['path']
    if path.exists():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
        return {'arm': arm, 'path': row['path'], 'status': 'verified-existing'}
    if not hasattr(local, 'clients'):
        local.clients = {}
    if arm not in local.clients:
        local.clients[arm] = KaggleOutput('pranavagrawai/tycho-dfranzen-wave-2-'+arm, 1)
    try:
        for attempt in range(4):
            time.sleep(1)
            try:
                raw = local.clients[arm].fetch(row['path'])
                break
            except Exception as error:
                response = getattr(error, 'response', None)
                if response is None or response.status_code != 429 or attempt == 3:
                    raise
                time.sleep(min(60, int(response.headers.get('Retry-After', 60))))
        assert hashlib.sha256(raw).hexdigest() == row['sha256']
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return {'arm': arm, 'path': row['path'], 'status': 'downloaded', 'bytes': len(raw)}
    except Exception as error:
        return {'arm': arm, 'path': row['path'], 'status': 'failed', 'error_type': type(error).__name__}
with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
    rows = list(pool.map(fetch, tasks))
with (ROOT / 'retrieval.jsonl').open('a') as stream:
    for row in rows:
        stream.write(json.dumps(row)+'\n')
from collections import Counter
print(json.dumps({'files':len(rows), 'statuses':dict(Counter(r['status'] for r in rows)), 'errors':[r for r in rows if r['status']=='failed']}))
