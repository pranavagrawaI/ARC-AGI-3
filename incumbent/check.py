"""Check that an arm carries every retained piece of the incumbent stack, or says why not.

Before launch:  python3 incumbent/check.py NOTEBOOK.ipynb [--without ID ...]
After a run:    python3 incumbent/check.py NOTEBOOK.ipynb --logs OUTPUT_DIR [--without ID ...]

A piece counts as carried only if all its notebook anchors appear in the code cells (and, with
--logs, all its log markers appear in some *.log under OUTPUT_DIR). Pieces in status 'retained'
or 'probation' must be carried unless named with --without; the card should say why. This is the
check that would have caught REAP448 dropping out in the SGLang migration.
"""
import argparse
import json
import sys
from pathlib import Path

STACK = Path(__file__).resolve().parent / 'stack.json'
REQUIRED = {'retained', 'probation'}


def code_text(notebook):
    cells = json.loads(Path(notebook).read_text())['cells']
    return '\n'.join(''.join(c['source']) for c in cells if c['cell_type'] == 'code')


def log_text(directory):
    return '\n'.join(p.read_text(errors='replace') for p in sorted(Path(directory).rglob('*.log')))


def check(notebook, logs=None, without=()):
    pieces = json.loads(STACK.read_text())['pieces']
    unknown = set(without) - {p['id'] for p in pieces}
    if unknown:
        raise SystemExit(f'unknown piece ids: {sorted(unknown)}')
    code = code_text(notebook)
    logged = log_text(logs) if logs else None
    rows, failed = [], False
    for piece in pieces:
        missing = [a for a in piece['notebook_anchors'] if a not in code]
        if logged is not None:
            missing += [f'log: {m}' for m in piece['log_markers'] if m not in logged]
        carried = not missing
        required = piece['status'] in REQUIRED and piece['id'] not in without
        if required and not carried:
            failed = True
        if piece['id'] in without and carried:
            failed = True
            missing = ['declared --without but the arm carries it']
        rows.append((piece['id'], piece['status'], 'carried' if carried else 'ABSENT', missing))
    return rows, failed


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('notebook')
    parser.add_argument('--logs')
    parser.add_argument('--without', action='append', default=[])
    args = parser.parse_args()
    rows, failed = check(args.notebook, args.logs, args.without)
    for piece, status, state, missing in rows:
        print(f'{piece:18} {status:13} {state:8} {"; ".join(missing)}')
    print('FAIL: a retained piece is missing or a declared removal is still present' if failed else 'OK')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
