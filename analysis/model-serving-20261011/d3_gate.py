"""Gate for hidden draw D3: pass only if the wave-2 b1-c14 qualifying pass is valid (wave-2/EXPERIMENT-CARD.md).
Exit 0 and write wave-2/evidence/GO-d3.json when valid; exit 1 otherwise. Idempotent; reads local files only.
  python3 -B d3_gate.py
"""
import json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
import os
RUN = Path(os.environ.get('D3_GATE_RUN', HERE / 'wave-2/b1-c14'))
FLOOR = 95


def main():
    reasons = []
    hist = RUN / 'status-history.jsonl'
    status = json.loads(hist.read_text().splitlines()[-1])['status'] if hist.exists() else 'no status'
    if 'COMPLETE' not in status:
        reasons.append(f'status {status}')
    out = RUN / 'output'
    if not (out / 'benchmark.json').exists():
        reasons.append('no benchmark.json')
    else:
        if not (RUN / 'readout.json').exists():
            subprocess.run([sys.executable, '-B', str(HERE / 'analyze.py'), str(out), 'B1C14'], check=True,
                           stdout=subprocess.DEVNULL)
        r = json.loads((RUN / 'readout.json').read_text())
        m = {k: v for k, v in r['markers'].items() if v}
        need = {'M2 B1 installed': 'M2 B1 installed', 'priority gate active': 'priority gate active: 14 concurrent streams',
                'READY after': 'READY after'}
        for key, want in need.items():
            if not any(want in x for k, v in m.items() if key in k for x in v):
                reasons.append(f'missing marker {want!r}')
        if any('exited during startup' in k for k in m):
            reasons.append('server exited during startup')
        if r['n_games'] != 25:
            reasons.append(f"{r['n_games']} games")
        if r['levels']['B1C14'] < FLOOR:
            reasons.append(f"levels {r['levels']['B1C14']} < {FLOOR}")
    verdict = {'valid': not reasons, 'reasons': reasons, 'status': status}
    (RUN / 'evidence').mkdir(exist_ok=True)
    if not reasons:
        launch = json.loads((RUN / 'launch-response.json').read_text())
        verdict.update(kernel=launch['ref'].split('/code/')[-1], version=launch['versionNumber'])
        (RUN / 'evidence/GO-d3.json').write_text(json.dumps(verdict, indent=1) + '\n')
    print(json.dumps(verdict))
    sys.exit(0 if not reasons else 1)


if __name__ == '__main__':
    main()
