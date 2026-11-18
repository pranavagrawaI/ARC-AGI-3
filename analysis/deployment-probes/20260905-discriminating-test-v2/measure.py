"""Independent full-pixel comparisons, not supplied to the model."""
import json
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent


def differences(before, after):
    groups = defaultdict(list)
    for r, row in enumerate(before):
        for c, value in enumerate(row):
            if value != after[r][c]:
                groups[(value, after[r][c])].append([r, c])
    return [{"from": key[0], "to": key[1], "count": len(cells), "cells": cells}
            for key, cells in groups.items()]


def main():
    source = json.loads((OUT / 'INPUT.json').read_text())
    results = {'up': differences(source['before_up'], source['after_up'])}
    for arm in ('a', 'b'):
        path = OUT / f'{arm}-TRANSITION.json'
        if path.exists():
            transition = json.loads(path.read_text())
            results[arm] = {'action': transition['action'],
                            'selected_action_diff': differences(transition['before'], transition['after']),
                            'initial_to_final_diff': differences(source['before_up'], transition['after'])}
    (OUT / 'MEASUREMENTS.json').write_text(json.dumps(results, indent=2) + '\n')
    for label, comparisons in results.items():
        if label == 'up':
            comparisons = {'diff': comparisons}
        print(label)
        for kind, groups in comparisons.items():
            if isinstance(groups, str):
                print(kind, groups)
                continue
            for group in groups:
                cells = group['cells']
                print(kind, group['from'], '->', group['to'], group['count'],
                      'bbox', [min(r for r,c in cells), max(r for r,c in cells), min(c for r,c in cells), max(c for r,c in cells)])


if __name__ == '__main__':
    main()
