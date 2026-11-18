"""Wait for the existing collector, then audit/plot once. Never launches inference."""
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
deadline = time.monotonic() + 11000
while time.monotonic() < deadline:
    marker = HERE / 'evidence/collection.json'
    if marker.exists():
        try:
            collected = json.loads(marker.read_text())
        except ValueError:
            time.sleep(1)
            continue
        snapshots = list((HERE / 'output').rglob('continuations/results.json'))
        if len(snapshots) == 1:
            command = [sys.executable, str(HERE / 'analyze.py'), '--episodes',
                       str(snapshots[0].parent / 'episodes'), '--plot']
            with (HERE / 'evidence/postprocess.log').open('x') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            print('Postprocess exit code:', result.returncode, flush=True)
            sys.exit(result.returncode)
        print('No unique episode matrix collected; inspect terminal logs.', flush=True)
        sys.exit(2)
    time.sleep(30)
raise SystemExit('Collection deadline reached; no inference launched or retried.')
