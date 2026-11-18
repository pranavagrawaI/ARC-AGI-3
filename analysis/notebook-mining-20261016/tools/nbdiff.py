"""Diff a dfranzen-derived notebook against dfranzen's: harness patch split per file, other code cells by index/heading.
usage: python3 -I -B nbdiff.py BASE.ipynb OTHER.ipynb OUT.diff  -> prints summary line; writes full unified diff
Read-only: parses JSON, never executes notebook content."""
import json, sys, re, difflib
def cells(p):
    nb = json.load(open(p)); out = []
    for c in nb['cells']:
        out.append((c['cell_type'], ''.join(c['source'])))
    return out
def split_patch(s):
    files = {}; cur = None
    for line in s.splitlines(keepends=True):
        m = re.match(r'diff --git a/(\S+) b/', line)
        if m: cur = m.group(1); files[cur] = []
        if cur: files[cur].append(line)
    return {k: ''.join(v) for k, v in files.items()}
def find_patch(cs):
    for t, s in cs:
        if t == 'code' and 'diff --git' in s[:2000] and len(s) > 50000: return s
    return ''
def code_cells(cs):
    return [s for t, s in cs if t == 'code' and not ('diff --git' in s[:2000] and len(s) > 50000)]
base, other, out = sys.argv[1:4]
B, O = cells(base), cells(other)
pb, po = split_patch(find_patch(B)), split_patch(find_patch(O))
rep = []; summ = []
for f in sorted(set(pb) | set(po)):
    a, b = pb.get(f, ''), po.get(f, '')
    if a != b:
        d = list(difflib.unified_diff(a.splitlines(), b.splitlines(), 'base/' + f, 'other/' + f, lineterm='', n=2))
        n = sum(1 for l in d if l[:1] in '+-' and l[:3] not in ('+++', '---'))
        summ.append(f'patch:{f}:{n}'); rep += d
cb, co = code_cells(B), code_cells(O)
sm = difflib.SequenceMatcher(None, [s[:300] for s in cb], [s[:300] for s in co])
pairs = []
for tag, i1, i2, j1, j2 in sm.get_opcodes():
    if tag == 'equal' or tag == 'replace':
        for k in range(max(i2 - i1, j2 - j1)):
            pairs.append((cb[i1 + k] if i1 + k < i2 else '', co[j1 + k] if j1 + k < j2 else ''))
    elif tag == 'delete': pairs += [(cb[k], '') for k in range(i1, i2)]
    elif tag == 'insert': pairs += [('', co[k]) for k in range(j1, j2)]
for idx, (a, b) in enumerate(pairs):
    if a != b:
        d = list(difflib.unified_diff(a.splitlines(), b.splitlines(), f'base/cell{idx}', f'other/cell{idx}', lineterm='', n=2))
        n = sum(1 for l in d if l[:1] in '+-' and l[:3] not in ('+++', '---'))
        summ.append(f'cell{idx}:{n}'); rep += d
open(out, 'w').write('\n'.join(rep) + '\n')
print(f'patchfiles base={len(pb)} other={len(po)} changes=' + ' '.join(summ))
