"""Build screen-2/c16: the screened F1x+REAP14 v1 notebook at 16 streams, nothing else changed. Never uploads.
python3 -B build_c16.py   (refuses to rebuild once launch-attempt.json exists)"""
import hashlib, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE / 'screen-1/merge/kernel/m2-f1x-reap14-screen1.ipynb'
PARENT_SHA = '9d634d6accb024d53b875106c4d77ad12948d895ea485cd4be44b7b4f83f56c0'  # drawn as D16 (57622267)
RUN, SLUG, TITLE = 'screen-2/c16', 'm2-f1x-reap16-screen2', 'M2 F1x REAP16 screen2'
SWAPS = [
    ("    'ARC3_MAX_ACTIVE_STREAMS': 14,  # F1x+REAP: was 10", "    'ARC3_MAX_ACTIVE_STREAMS': 16,  # c16: was 14"),
    ('    MAXREQ=14,  # F1x+REAP', '    MAXREQ=16,  # c16 (was 14)  # F1x+REAP'),
    ('    CUDAGRAPH_MAXBS=14,  # F1x+REAP', '    CUDAGRAPH_MAXBS=16,  # c16 (was 14)  # F1x+REAP'),
    ('    MAMBA_CACHE=84,  # F1x+REAP: 6 slots per stream', '    MAMBA_CACHE=96,  # c16 (was 84)  # F1x+REAP: 6 slots per stream'),
    ('served by REAP448 (448 target experts, full 512-expert MTP drafter) at 14 streams.',
     'served by REAP448 (448 target experts, full 512-expert MTP drafter) at 16 streams (screen-2 c16).'),
]


def sha(b): return hashlib.sha256(b).hexdigest()


def build():
    raw = PARENT.read_bytes()
    assert sha(raw) == PARENT_SHA, 'parent drifted from the drawn screen-1 v1'
    nb = json.loads(raw)
    hits = {old: 0 for old, _ in SWAPS}
    for c in nb['cells']:
        text = ''.join(c['source'])
        for old, new in SWAPS:
            hits[old] += text.count(old)
            text = text.replace(old, new)
        c['source'] = text.splitlines(True)
    assert all(n == 1 for n in hits.values()), hits
    return nb


def main():
    d = HERE / RUN
    if (d / 'launch-attempt.json').exists():
        print(f'{RUN} launched; not rebuilding'); return
    k = d / 'kernel'; k.mkdir(parents=True, exist_ok=True)
    (k / f'{SLUG}.ipynb').write_text(json.dumps(build(), indent=1) + '\n')
    meta = json.loads((PARENT.parent / 'kernel-metadata.json').read_text())
    meta.update(id=f'pranavagrawai/{SLUG}', title=TITLE, code_file=f'{SLUG}.ipynb', is_private=True)
    (k / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    runs = json.loads((HERE / 'runs.json').read_text())
    runs['runs'] = [r for r in runs['runs'] if r['run'] != RUN] + [
        {'run': RUN, 'slug': SLUG, 'parent_sha256': PARENT_SHA,
         'notebook_sha256': sha((k / f'{SLUG}.ipynb').read_bytes()), 'metadata': meta}]
    (HERE / 'runs.json').write_text(json.dumps(runs, indent=2) + '\n')
    print(json.dumps({'run': RUN, 'sha256': runs['runs'][-1]['notebook_sha256']}))


if __name__ == '__main__':
    main()
