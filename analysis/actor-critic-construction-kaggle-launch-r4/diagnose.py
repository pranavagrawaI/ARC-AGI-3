"""Read-only R3 diagnosis; no provider calls, package edits or game substitution."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PUBLIC = ROOT / 'analysis/deployment-probes/20260901T054322Z-public25-kaggle-v1/public-environment-manifest.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads(PUBLIC.read_text())
    records = []
    for record in manifest['files']:
        if '/ls20/' in record['path']:
            local = ROOT / record['path']
            records.append({**record, 'local_bytes': local.stat().st_size,
                            'local_sha256': sha(local), 'matches': sha(local) == record['sha256']})
    copies = []
    for directory, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in {'.git', '.venv', '__pycache__', 'uv-cache', 'node_modules', '.cache', 'site-packages'}]
        if 'ls20.py' in files:
            source = Path(directory) / 'ls20.py'
            copies.append({'path': str(source.relative_to(ROOT)), 'sha256': sha(source)})
    # Real Arcade in a fresh scrubbed interpreter. The mounted-shaped positive
    # control is explicitly the nonmatching checkout game, NOT the public asset.
    with tempfile.TemporaryDirectory(prefix='r3-game-diagnosis-') as temporary:
        tmp = Path(temporary)
        mounted = tmp / 'kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files'
        import shutil
        shutil.copytree(ROOT / 'environment_files', mounted)
        program = '''import json,sys
from pathlib import Path
from arc_agi import Arcade,OperationMode
root=Path(sys.argv[1])
rows=[]
for relative in ['kaggle/input/arc-prize-2026-arc-agi-3/environment_files','kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files']:
    arcade=Arcade(operation_mode=OperationMode.OFFLINE,environments_dir=str(root/relative),recordings_dir=str(root/'recordings'))
    env=arcade.make('ls20')
    row={'path':relative,'discovered':[e.game_id for e in arcade.get_environments()],'make_none':env is None}
    try:
        frame=env.reset()
        row['reset_game_id']=frame.game_id
    except Exception as error:
        row['error']=f'{type(error).__name__}: {error}'
    rows.append(row)
(root/'discovery.json').write_text(json.dumps(rows))
'''
        run = subprocess.run([sys.executable, '-I', '-B', '-c', program, str(tmp)],
                             cwd=tmp, env={'PATH': '/opt/bin:/usr/local/bin:/usr/bin:/bin', 'HOME': str(tmp), 'LANG': 'C.UTF-8', 'ARC_DISABLE_DOTENV': '1'},
                             text=True, capture_output=True, timeout=30, check=True)
        discovery = json.loads((tmp / 'discovery.json').read_text())
        assert discovery[0]['make_none'] and 'NoneType' in discovery[0]['error']
        assert discovery[1]['reset_game_id'] == 'ls20-9607627b'
    from arc_agi import Arcade
    library = Path(inspect.getfile(Arcade))
    protected = json.loads((ROOT / 'analysis/actor-critic-construction-openrouter-20260914T154902Z/PRELAUNCH-MANIFEST.json').read_text())['sources']
    protected_drift = [name for name, digest in protected.items() if sha(ROOT / name) != digest]
    assert len(protected) == 422 and not protected_drift
    result = {'status': 'HOLD', 'reason': 'Exact Kaggle public LS20 bytes unavailable locally; local game differs from public manifest.',
              'provider_calls': 0, 'launches': 0, 'r4_package_created': False,
              'public_manifest_sha256': sha(PUBLIC), 'public_records': records,
              'local_source_copies': copies, 'arcade_implementation': str(library), 'arcade_implementation_sha256': sha(library),
              'discovery_scope': 'Real Arcade/reset, scrubbed environment, temporary Kaggle-shaped directory layout; positive control uses NONMATCHING local game, no mounted public parity or worker claim.',
              'discovery': discovery, 'stdout': run.stdout, 'stderr': run.stderr,
              'protected_sources_verified': len(protected), 'protected_drift': protected_drift}
    (HERE / 'DIAGNOSIS.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
