"""Classify dfranzen-family notebooks by rerun-relevant deltas. Read-only JSON parsing.
usage: python3 -I -B classify.py BASE.ipynb PULLED_DIR SCORES.jsonl... > family.jsonl"""
import json, sys, re, os, glob, difflib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
base, pulled, *score_files = sys.argv[1:]
scores = {}
for f in score_files:
    for l in open(f):
        d = json.loads(l); scores[d['ref']] = d
def cells(p):
    return [(c['cell_type'], ''.join(c['source'])) for c in json.load(open(p))['cells']]
def patch_of(cs):
    for t, s in cs:
        if t == 'code' and 'diff --git' in s[:2000] and len(s) > 50000: return s
    return None
def code(cs):
    return '\n'.join(s for t, s in cs if t == 'code' and not ('diff --git' in s[:2000] and len(s) > 50000))
B = cells(base); bp = patch_of(B); bcode = code(B).splitlines()
for d in sorted(glob.glob(os.path.join(pulled, '*/'))):
    nbs = glob.glob(d + '*.ipynb')
    if not nbs: continue
    meta = json.load(open(d + 'kernel-metadata.json'))
    ref = meta['id']
    try: O = cells(nbs[0])
    except Exception as e: continue
    op = patch_of(O)
    fam = op is not None
    added = [l[1:] for l in difflib.unified_diff(bcode, code(O).splitlines(), lineterm='', n=0) if l.startswith('+') and not l.startswith('+++')]
    removed = [l[1:] for l in difflib.unified_diff(bcode, code(O).splitlines(), lineterm='', n=0) if l.startswith('-') and not l.startswith('---')]
    env = sorted(set(m.group(1) + '=' + m.group(2).strip().rstrip(',') for l in added
                     for m in [re.search(r"""['"]((?:ARC3|LOCAL_ANALYZER|EXPOSE)_[A-Z0-9_]+)['"]\s*[:\]]\s*=?\s*(.+)""", l)] if m))
    cfg = sorted(set(l.strip() for l in added if re.search(r'\b(MAXREQ|CUDAGRAPH_MAXBS|MAMBA_CACHE|MEMFRAC|CTX|CFG_OVERRIDE|concurrency|max_runtime_s_per_game|n_passes)\b', l)))
    sc = scores.get(ref, {})
    out = dict(ref=ref, public=sc.get('public'), best=sc.get('best'), version=sc.get('version'),
               family=fam, patch_identical=(op == bp) if fam else None,
               dprime=any('install_d' in l for l in added), extra_patch=any('git apply' in l or '_PATCH_B64' in l or 'patch' in l and 'subprocess' in l for l in added),
               n_added=len(added), n_removed=len(removed), env=env, cfg=cfg,
               models=meta.get('model_sources'), datasets=meta.get('dataset_sources'))
    print(json.dumps(out))
