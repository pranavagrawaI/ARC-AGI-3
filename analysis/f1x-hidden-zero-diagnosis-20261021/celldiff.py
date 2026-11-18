"""Cell-by-cell unified diff of two notebooks' code (skips the huge patch/overlay cells unless they differ by hash)."""
import json, sys, difflib, hashlib
def cells(p): return [''.join(c['source']) for c in json.load(open(p))['cells'] if c['cell_type']=='code']
a,b=cells(sys.argv[1]),cells(sys.argv[2])
print(len(a),len(b))
for i,(x,y) in enumerate(zip(a,b)):
    if x==y: continue
    if max(len(x),len(y))>30000:
        print(f'### cell {i}: LARGE differs sha {hashlib.sha256(x.encode()).hexdigest()[:12]} vs {hashlib.sha256(y.encode()).hexdigest()[:12]} len {len(x)} vs {len(y)}'); continue
    print(f'### cell {i}')
    sys.stdout.writelines(difflib.unified_diff(x.splitlines(1),y.splitlines(1),n=1))
