"""Fetch per-arm session files by the exact paths the driver recorded (games[*].sessions). Read-only, no listing."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import collect_exact_paths as cep  # noqa: E402


def main(game):
    base = cep.BASE
    result = json.loads((base / 'runtime-evidence-exact-request/model-runtime/official-launcher-result.json').read_text())
    files = []
    for g in result['games']:
        ws = f"model-runtime/solver-workspaces/{g['workspace']}"
        for s in g['sessions']:
            for name in s['ipc_files']:
                files.append(f"{ws}/{s['session_dir']}/ipc/{name}")
            for name in s['files']:
                files.append(f"{ws}/{s['session_dir']}/{name}")
            for rel in s['host_events']:
                files.append(f"{ws}/{rel}")
                files.append(f"{ws}/{rel.rsplit('/', 1)[0]}/FROZEN.json" if '/' in rel else f"{ws}/FROZEN.json")
            for rel in s['constructor_results']:
                files.append(f"{ws}/{rel}")
        for name in g['official']['recording_files']:
            files.append(f"{ws}/recordings/{name}")
    files = list(dict.fromkeys(files))
    print(json.dumps({'game': game, 'planned_files': len(files)}), file=sys.stderr)
    cep.main(files, f'sessions-{game}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'cd82')
