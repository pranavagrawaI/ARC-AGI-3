"""Snapshot research-document identities and the board without modifying evidence.

Run once with a fresh destination: python collect_sources.py OUTPUT_DIRECTORY
The inventory is discovery coverage, not a claim that every file was adjudicated.
"""
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = Path(sys.argv[1]).resolve()
DEST.mkdir(parents=True, exist_ok=False)

def output(name, obj):
    with (DEST / name).open('x') as stream:
        json.dump(obj, stream, indent=2)
        stream.write('\n')

paths = subprocess.check_output(['rg', '--files', '--hidden', '-g', '!.git',
    'analysis', 'experiments', 'docs', 'solvers', 'deployment', 'Questions and Ideas'],
    cwd=ROOT, text=True).splitlines()
records = []
for relative in sorted(paths):
    path = ROOT / relative
    if relative.startswith('analysis/what-we-learned-20260909/'):
        continue
    if path.suffix.lower() != '.md':
        continue
    if any(part in {'third_party', 'site-packages', '.venv', 'node_modules'} for part in path.parts):
        continue
    data = path.read_bytes()
    records.append({'path': relative, 'bytes': len(data),
                    'sha256': hashlib.sha256(data).hexdigest(),
                    'mtime_utc': datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()})
output('source-documents.json', records)
cursor = 0
events = []
pages = []
while True:
    page = json.loads(subprocess.check_output(['python3', '/home/pranav/.agent-board/board.py',
          'log', '--since', str(cursor), '--limit', '100', '--json'], text=True))
    events.extend(page['events'])
    pages.append({k: page[k] for k in ['next_cursor', 'head', 'has_more']})
    if not page['has_more']:
        break
    if page['next_cursor'] <= cursor:
        raise RuntimeError('Board cursor did not advance')
    cursor = page['next_cursor']
output('board-log.json', {'pages': pages, 'events': events, 'next_cursor': page['next_cursor']})
output('snapshot.json', {'captured_utc': datetime.now(timezone.utc).isoformat(),
    'workspace': str(ROOT),
    'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    'git_status': subprocess.check_output(['git', 'status', '--short', '--branch'], cwd=ROOT, text=True),
    'document_count': len(records), 'board_events': len(events),
    'limits': 'Document inventory excludes vendored dependencies and this report. Discovery is not substantive review. Board includes unrelated threads for cursor continuity; report cites only project evidence. Raw runs remain in original locations.'})
print(json.dumps({'documents': len(records), 'board_events': len(events), 'next_cursor': page['next_cursor']}))
