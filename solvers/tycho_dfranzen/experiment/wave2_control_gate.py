"""wave-2 control gate: launch control once both lean arms are collected and pass the card's
mechanical abort checks; otherwise record a hold. One launch attempt, never a retry.

Checks per lean arm (EXPERIMENT-CARD.md, Stop): critical outputs collected, benchmark present,
no crashed game, no GenerationError (missing action), Actor-only roles, projection window 65536.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
LEAN_ARMS = ('lean_cap4', 'lean_prune')
WINDOW = 64 * 1024


def arm_problems(root):
    """Card abort reasons for one collected lean arm; empty means clear."""
    receipt = root/'collection-receipt.json'
    if not receipt.exists() or not json.loads(receipt.read_text())['critical_complete']:
        return ['critical outputs not collected']
    job = root/'output/tycho-dfranzen'
    if not (job/'benchmark.json').exists():
        return ['no benchmark.json (failed before gameplay)']
    problems = []
    for run in json.loads((job/'benchmark.json').read_text())['game_runs']:
        note = str(run.get('solver_note') or '')
        if run['state'] == 'crashed' or 'GenerationError' in note:
            problems.append(f"{run['game_id']}: {run['state']} {note[:120]}")
    games = sorted((job/'hybrid-games').glob('*/compute.json'))
    if not games:
        problems.append('no per-game compute.json')
    for compute in games:
        roles = set(json.loads(compute.read_text())['roles'])
        if roles - {'actor'}:
            problems.append(f'{compute.parent.name}: non-Actor roles {sorted(roles)}')
        projection = compute.parent/'context-projection.jsonl'
        windows = {json.loads(line).get('window') for line in projection.read_text().splitlines()} if projection.exists() else set()
        if windows != {WINDOW}:
            problems.append(f'{compute.parent.name}: projection windows {sorted(map(str, windows))}')
    return problems


def gate(directory):
    """('launch', {}) when both lean arms are clear, ('hold', problems) otherwise, None if pending."""
    if not all((directory/arm/'collection-receipt.json').exists() for arm in LEAN_ARMS):
        return None
    problems = {arm: arm_problems(directory/arm) for arm in LEAN_ARMS}
    problems = {arm: found for arm, found in problems.items() if found}
    return ('hold', problems) if problems else ('launch', {})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--hours', type=float, default=14)
    args = parser.parse_args()
    until = time.monotonic() + args.hours*3600
    while (decision := gate(args.directory)) is None and time.monotonic() < until:
        time.sleep(60)
    record = {'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'decision': decision[0] if decision else 'timeout', 'problems': decision[1] if decision else {}}
    (args.directory/'control-gate.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record), flush=True)
    if decision and decision[0] == 'launch':
        from kaggle.api.kaggle_api_extended import KaggleApi
        import wave_run
        api = KaggleApi(); api.authenticate()
        wave_run.launch(args.directory, api, HERE/'budget.json', ['control'])


if __name__ == '__main__':
    main()
