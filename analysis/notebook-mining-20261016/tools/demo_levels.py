"""Levels per game in 10-game / 25-min dfranzen-family demo runs (events artifacts). usage: python3 -I -B demo_levels.py DIR..."""
import json, glob, os, sys
G = ['ar25', 'ft09', 'lp85', 'r11l', 're86', 'sb26', 'sc25', 'tr87', 'tu93', 'vc33']
print('run'.ljust(58), ' '.join(g for g in G), ' total actions')
for d in sys.argv[1:]:
    lv = {}; acts = 0
    for f in glob.glob(os.path.join(d, '**', '*_p0_events.jsonl'), recursive=True):
        g = os.path.basename(f)[:4]; m = 0
        for line in open(f):
            e = json.loads(line)
            if e.get('type') != 'action': continue
            acts += 1; m = max(m, int(e.get('score') or 0))
        lv[g] = m
    if not lv: continue
    extra = sorted(set(lv) - set(G))
    print(os.path.basename(d.rstrip('/'))[:58].ljust(58), ' '.join(f'{lv.get(g, "-"):>4}' for g in G), f'{sum(lv.get(g, 0) for g in G):5d} {acts:6d}', ('EXTRA ' + str(extra)) if extra else '')
