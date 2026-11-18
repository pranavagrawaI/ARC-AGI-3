"""Freeze a private Kaggle diagnostic notebook, original serving cells unchanged."""
from pathlib import Path
import base64,hashlib,json,random,zlib

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SLUG='counterexample-continuations-v2-20261013'
DONOR=ROOT/'analysis/m2-winners-20261001/audit/dfranzen-repo'
CONTROL=ROOT/'analysis/m2-winners-20261001/runs/control'

def sha(b):return hashlib.sha256(b).hexdigest()
def cell(s):return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':s.splitlines(True)}

def main():
    if (HERE/'evidence/launch-attempt.json').exists():raise RuntimeError('already attempted')
    assert json.loads((HERE/'evidence/e2e-qualification.json').read_text())['cpu_fixtures_only']
    for name in ['b1','control']:
        assert json.loads((HERE/f'evidence/restore-{name}-v4/checkpoint.json').read_text())['qualified']
    matrix=[]
    for checkpoint in ['b1','control']:
        for seed in [1701,2903]:
            arms=['control','reminder','counterexample'];random.Random(seed).shuffle(arms)
            for arm in arms:matrix.append({'id':f'{checkpoint}-{seed}-{arm}','checkpoint':checkpoint,'seed':seed,'arm':arm})
    (HERE/'matrix.json').write_text(json.dumps(matrix,indent=2)+'\n')
    files={name:(HERE/name).read_text() for name in ['restore.py','worker.py','run_matrix.py','matrix.json']}
    original_inputs={str(path.relative_to(HERE)):path.read_text() for path in (HERE/'inputs').glob('*.json')}
    files.update(original_inputs)
    # Freeze complete importable trees; the diagnostic worker never consumes fixture routes or future frames.
    source_roots={'b1':ROOT/'solvers/m2_b1/src','control':DONOR/'ARC3-Inference','taaf':DONOR/'tufa-arc-agi-framework/src'}
    for name,root in source_roots.items():
        for path in root.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix in ['.py','.json','.yaml','.yml','.j2','.html','.css','.txt']:
                files[f'sources/{name}/{path.relative_to(root)}']=path.read_text()
    cards={}
    for episode in matrix:
        card=f"""# {SLUG}/{episode['id']}
Question: Does {episode['arm']} improve next-level completion at checkpoint {episode['checkpoint']}?
Posture: Matched diagnostic continuation; selected cases, not general efficacy.
Contrast: Same checkpoint and seed {episode['seed']} across control, reminder and counterexample; matched original runtime per checkpoint.
Prediction: Counterexample may improve B1 completion; the already-self-repairing control tests unnecessary-feedback cost.
Measure: result.json levels_completed>=2, game score including the historical action prefix; wire.jsonl raw usage; actions.json; checkpoint.json restoration audit. Prediction and repair behavior are secondary transcript evidence.
Cap: 32768 newly generated tokens; 32 model requests; 120 absolute seconds per request; 900 absolute seconds after branch; one worker; no retry; no monetary endpoint charges (self-hosted Kaggle). GPU session shares the matrix's 10800-second hard cap, including setup. Matrix maximum 393216 generated tokens, 12 episodes, one RTX PRO6000.
Stop: Stop on L2 completion or any cap. Entire matrix stops on restoration/integrity failure, missing usage, HTTP/provider failure, request deadline or unresolved request exposure. A normal episode cap ends this episode only. Preserve all attempted episodes.
Invalid if: Checkpoint source/pixels/action prefix/helper replay diverges; future data reaches model; usage absent; unequal arm budgets; model/quantization/serving differs from archived notebook; treatment additions differ from frozen feedback.
Result map: >=2 paired completion gains versus control and no paired loss across 4 blocks -> qualify generic triggering next; reminder tie -> prefer reminder; repair without score -> inspect residual defect/search bound; all fail -> inspect cap/repair before rejecting approach; self-repair case harmed -> tighten trigger. No full-evaluation promotion from this screen.
Outputs: episodes/{episode['id']}/ and episodes/{episode['id']}.log.
Authority: Pranav's 'We should give it a try' following the proposed bounded continuation comparison. Private diagnostic run only; no competition submission.
Packaging revision: predecessor SaveKernel returned HTTP400; read-only reconciliation found no remote run, no active session and unchanged quota. This new run ID uses lossless long-range compression; actor source, observations, contrasts and compute caps are unchanged. The original launch attempt remains preserved.
Common continuation setup: isolated sequential episodes, priority gate disabled; exact archived first request plus assigned feedback, original sampling with paired seed. Clocks restart with the fixed continuation budget. Initial history PNGs use original bytes. Restoration comparisons normalize only pixel-equivalent PNG encodings, printed set ordering, and named elapsed/remaining clock fields.
"""
        cards[f'cards/{episode["id"]}.md']=card
    files.update(cards)
    for name,source in cards.items():
        p=HERE/name;p.parent.mkdir(exist_ok=True);p.write_text(source)
    manifest={name:sha(source.encode()) for name,source in files.items()}
    files['package-manifest.json']=json.dumps(manifest,indent=2)
    # Entire request snapshots are repetitive too; long-range compression captures those repeats.
    import lzma
    packed=base64.b64encode(lzma.compress(json.dumps(files).encode(),preset=9)).decode()
    assert json.loads(lzma.decompress(base64.b64decode(packed)))==files
    install=f"""# COUNTEREXAMPLE CONTINUATIONS: frozen past-only inputs and source trees.
import base64,hashlib,json,lzma
CONTINUATION_DIR=WORKING_DIR/'continuations'
CONTINUATION_DIR.mkdir(exist_ok=True)
_files=json.loads(lzma.decompress(base64.b64decode({packed!r})))
for _name,_text in _files.items():
    _path=CONTINUATION_DIR/_name
    _path.parent.mkdir(parents=True,exist_ok=True)
    _path.write_text(_text)
for _name,_hash in json.loads(_files['package-manifest.json']).items():
    assert hashlib.sha256((CONTINUATION_DIR/_name).read_bytes()).hexdigest()==_hash
print('Frozen continuation package verified',len(_files),'files')
"""
    run="""# COUNTEREXAMPLE CONTINUATIONS: execute once; no benchmark/submission path.
if TRUE_SUBMISSION:raise RuntimeError('Diagnostic notebook cannot run as competition submission')
competition_env_files='/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files'
assert Path(competition_env_files).is_dir()
subprocess.run([sys.executable,'-B',str(CONTINUATION_DIR/'run_matrix.py'),competition_env_files,SERVER_BASE_URL],check=True)
print((CONTINUATION_DIR/'results.json').read_text())
"""
    notebook=json.loads((CONTROL/'dfranzen-m2-control.ipynb').read_text())
    kept=[]
    for c in notebook['cells']:
        source=''.join(c['source'])
        if c['cell_type']!='code':continue
        if source.startswith('%%writefile /kaggle/harness-changes.patch\n'):
            patch=source.split('\n',1)[1].encode()
            patch_packed=base64.b64encode(lzma.compress(patch,preset=9)).decode()
            assert lzma.decompress(base64.b64decode(patch_packed))==patch
            kept.append(cell("from pathlib import Path\nimport base64,lzma,hashlib\n"
                f"_patch=lzma.decompress(base64.b64decode({patch_packed!r}))\n"
                f"assert hashlib.sha256(_patch).hexdigest()=={sha(patch)!r}\n"
                "Path('/kaggle/harness-changes.patch').write_bytes(_patch)\n"))
            continue
        if source.startswith('# Restore the deployment target') or source.startswith('# Make one-off changes') or source.startswith('from html import escape'):continue
        if source.startswith("print('Starting benchmark..."):
            kept.extend([cell(install),cell(run)]);continue
        if 'setup_env = {' in source:
            source += "\nif TRUE_SUBMISSION:raise RuntimeError('Continuation diagnostics cannot be submitted')\n"
        kept.append(cell(source))
    notebook['cells']=kept
    kernel=HERE/'kernel';kernel.mkdir(exist_ok=True)
    nb=kernel/(SLUG+'.ipynb');nb.write_text(json.dumps(notebook,indent=1)+'\n')
    meta=json.loads((CONTROL/'kernel-metadata.json').read_text());meta.update(id='pranavagrawai/'+SLUG,title='Counterexample Continuations V2 20261013',code_file=nb.name,is_private=True)
    (kernel/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    identity={'notebook_sha256':sha(nb.read_bytes()),'metadata':meta,'manifest':manifest,'hard_cap_seconds':10800,
              'episodes':12,'generated_token_cap':393216,'source_notebook_sha256':sha((CONTROL/'dfranzen-m2-control.ipynb').read_bytes())}
    (HERE/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    print('built',nb,'bytes',nb.stat().st_size,'frozen files',len(files))

if __name__=='__main__':main()
