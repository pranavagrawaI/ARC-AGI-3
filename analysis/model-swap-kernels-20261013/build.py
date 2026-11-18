import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import nbformat
from IPython.core.interactiveshell import InteractiveShell

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
BASE=REPO/'analysis/model-serving-20261011/wave-1/e2-c14/kernel/wave1-e2-c14-20261011.ipynb'
SLUG='model-swap-tinfield-smoke-20261013'

def replace(nb,old,new):
    hits=[c for c in nb['cells'] if c['cell_type']=='code' and old in ''.join(c['source'])]
    assert len(hits)==1,(old,len(hits))
    s=''.join(hits[0]['source']);assert s.count(old)==1
    hits[0]['source']=s.replace(old,new).splitlines(True)

def build():
    repaired='--repaired' in sys.argv
    run=ROOT/'tinfield'/('repaired' if repaired else '')
    slug='model-swap-tinfield-smoke-r2-20261013' if repaired else SLUG
    assert not (run/'launch-attempt.json').exists(),'already frozen/launched'
    proof=json.loads((ROOT/'tinfield/evidence/ple-identity.json').read_text())
    assert proof['all_ple_tensors_identical']
    base=json.loads(BASE.read_text());nb=copy.deepcopy(base)
    old="MODEL_DIR         = '/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1'"
    source=(run/'assemble.py').read_text()
    new="INCUMBENT_MODEL_DIR = '/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1'\n"
    new+="TINFIELD_CORE_DIR = '/kaggle/input/models/natnitarach/tinfield-1-nvfp4-probe/pytorch/core/1'\n"
    new+=source+"\nMODEL_DIR = assemble(TINFIELD_CORE_DIR, INCUMBENT_MODEL_DIR, '/kaggle/tinfield-model-view')"
    replace(nb,old,new)
    replace(nb,'assert q.get("quant_method") == "auto-round" and q.get("bits") == 4, "MODEL_DIR must be the Intel/albucino AutoRound INT4 checkpoint"',
            'assert q.get("quant_method") == "modelopt" and q.get("quant_algo") == "NVFP4", "Tinfield target must be the public ModelOpt NVFP4 checkpoint"')
    replace(nb,'"--dtype", "bfloat16", "--quantization", "auto-round",','"--dtype", "bfloat16", "--quantization", "modelopt_fp4",')
    # Interactive-only smoke controls. Every TRUE_SUBMISSION body stays literal.
    replace(nb,'NOTEBOOK_START_EPOCH = time.time()',
            'NOTEBOOK_START_EPOCH = time.time()\nif not TRUE_SUBMISSION:\n    SERVER_STARTUP_TIMEOUT = 1300\n')
    replace(nb,"    if float(getattr(target, 'max_runtime_s', 0.0) or 0.0) != 32400.0:",
            "    bm.solver.max_runtime_s_per_game = 40*60  # interactive smoke only\n    if float(getattr(target, 'max_runtime_s', 0.0) or 0.0) != 32400.0:")
    replace(nb,'soft_end = datetime.fromtimestamp(NOTEBOOK_START_EPOCH) + timedelta(seconds=budget - min(600.0, budget / 2))',
            'soft_end = datetime.now() + timedelta(seconds=40*60)  # interactive play cap, after server startup')
    # The probe runs only in the smoke; submission requests remain unchanged.
    probe='''
if not TRUE_SUBMISSION:
    import math as _math
    from urllib.request import Request as _Request
    with urlopen(f'http://127.0.0.1:{SERVED_MODEL_PORT}/get_server_info', timeout=30) as _r:
        _info = json.loads(_r.read())
    (WORKING_DIR / 'model-swap-server-info.json').write_text(json.dumps(_info, indent=2))
    _payload = dict(model=SERVED_MODEL_NAME, prompt='Compute 2+2. Answer with the numeral only: ', max_tokens=32, temperature=0, logprobs=1)
    _req = _Request(SERVER_BASE_URL + '/completions', data=json.dumps(_payload).encode(), headers={'Content-Type':'application/json'})
    with urlopen(_req, timeout=180) as _r:
        _answer = json.loads(_r.read())
    (WORKING_DIR / 'model-swap-finite-answer.json').write_text(json.dumps(_answer, indent=2))
    _probs = _answer['choices'][0]['logprobs']['token_logprobs']
    assert _probs and all(_math.isfinite(float(_p)) for _p in _probs if _p is not None), 'nonfinite or empty token logprobs'
    print('MODEL_SWAP_FINITE_ANSWER', len(_probs), repr(_answer['choices'][0]['text']))
'''
    nb['cells'][12]['source']=(''.join(nb['cells'][12]['source'])+probe).splitlines(True)
    # Match mounted core format to SGLang; retain drafter, backend auto and all c14 settings.
    kd=run/'kernel';kd.mkdir(parents=True,exist_ok=True)
    notebook=kd/(slug+'.ipynb');notebook.write_text(json.dumps(nb,indent=1)+'\n')
    meta=json.loads(BASE.with_name('kernel-metadata.json').read_text())
    meta.update(id='pranavagrawai/'+slug,title=slug.replace('-',' '),code_file=notebook.name,is_private=True)
    meta['model_sources'].append('natnitarach/tinfield-1-nvfp4-probe/PyTorch/core/1')
    (kd/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    nbformat.validate(nbformat.read(notebook,as_version=4))
    shell=InteractiveShell.instance()
    for i,c in enumerate(nb['cells']):
        if c['cell_type']=='code':compile(shell.transform_cell(''.join(c['source'])),f'cell-{i}','exec',flags=0x2000)
    # Full literal comparison of every explicit submission branch.
    import ast
    def submission_bodies(n):
        out=[]
        for c in n['cells']:
            if c['cell_type']!='code':continue
            s=''.join(c['source'])
            if s.startswith('%%'):continue
            tree=ast.parse(shell.transform_cell(s))
            for node in ast.walk(tree):
                if isinstance(node,ast.If) and isinstance(node.test,ast.Name) and node.test.id=='TRUE_SUBMISSION':
                    out.append([ast.dump(x,include_attributes=False) for x in node.body])
        return out
    assert submission_bodies(nb)==submission_bodies(base)
    result=subprocess.run([sys.executable,'-B',str(REPO/'incumbent/check.py'),str(notebook),'--without','b1'],text=True,capture_output=True)
    (run/'incumbent-check.log').write_text(result.stdout+result.stderr);assert result.returncode==0
    changed=[i for i,(a,b) in enumerate(zip(base['cells'],nb['cells'])) if a['source']!=b['source']]
    receipt={'notebook':str(notebook.relative_to(REPO)),'base_sha256':hashlib.sha256(BASE.read_bytes()).hexdigest(),
             'notebook_sha256':hashlib.sha256(notebook.read_bytes()).hexdigest(),'metadata_sha256':hashlib.sha256((kd/'kernel-metadata.json').read_bytes()).hexdigest(),
             'changed_cells':changed,'submission_if_bodies_identical':True,'notebook_schema_valid':True,'all_code_cells_compile':True,
             'b1_exception':'Named exact c14 base lacks B1; weights-only/rerun contract forbids adding it. Board 684 records conflict; D3 failed in thread 682.'}
    (run/'LOCAL-QUALIFICATION.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

if __name__=='__main__':build()
