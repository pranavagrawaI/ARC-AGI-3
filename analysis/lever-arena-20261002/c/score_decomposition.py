"""Descriptive ceiling at observed completions; never a causal ablation.
Run from repository root. Source scores are rounded to two decimals, so all
headroom values inherit up to .005 mean rounding error. Full runs differ in time.
"""
import json
import re
from pathlib import Path
out = {}
for label in ('control', 'control-r2', 's1'):
    source = Path(f'analysis/m2-winners-20261001/runs/out-{label}/summary.txt')
    rows = []
    for game, score, completed, total, actions, tokens in re.findall(
        r'  (\S+): score=([\d.]+), levels=([\d.]+)/(\d+), actions=(\d+), tokens=(\d+)',
        source.read_text(),
    ):
        score, completed, total = float(score), int(float(completed)), int(total)
        ceiling = 100 * completed * (completed + 1) / (total * (total + 1))
        rows.append(dict(game=game, score=score, levels=completed,
                         total_levels=total, completion_ceiling=ceiling,
                         efficiency_headroom=ceiling-score))
    assert len(rows) == 25, (label, len(rows))
    out[label] = dict(source=str(source), games=len(rows),
                     mean_score=sum(r['score'] for r in rows)/len(rows),
                     mean_completion_ceiling=sum(r['completion_ceiling'] for r in rows)/len(rows),
                     mean_efficiency_headroom=sum(r['efficiency_headroom'] for r in rows)/len(rows),
                     rows=rows)
Path(__file__).with_name('score-decomposition.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps({k: {a:b for a,b in v.items() if a != 'rows'} for k,v in out.items()}, indent=2))
