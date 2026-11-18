"""Produce exact future session collection paths from retained launcher JSON. No network."""
import json
import sys
from pathlib import PurePosixPath,Path

def paths(result):
    output=[]
    def add(value):
        p=PurePosixPath(value)
        if p.is_absolute() or '..' in p.parts:raise ValueError('unsafe collection path')
        output.append(str(p))
    for game in result['games']:
        base=f"model-runtime/solver-workspaces/{game['workspace']}"
        for session in game['sessions']:
            for name in session['ipc_files']:add(f"{base}/{session['session_dir']}/ipc/{name}")
            for name in session['files']:add(f"{base}/{session['session_dir']}/{name}")
            for key in ('host_events','constructor_results','constructor_files'):
                for rel in session.get(key,[]):add(f'{base}/{rel}')
        for name in game['official']['recording_files']:add(f'{base}/recordings/{name}')
    return sorted(set(output))
if __name__=='__main__':
    print(json.dumps({'network_calls':0,'planned_paths':paths(json.loads(Path(sys.argv[1]).read_text()))},indent=2))
