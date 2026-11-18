"""Same local qualification as build.py, applied to the r3 candidate."""
import ast, hashlib, json, subprocess, sys
from pathlib import Path
import nbformat
from IPython.core.interactiveshell import InteractiveShell
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]; REPO=ROOT.parents[1]
BASE=REPO/'analysis/model-serving-20261011/wave-1/e2-c14/kernel/wave1-e2-c14-20261011.ipynb'
R2=ROOT/'tinfield/repaired/kernel/model-swap-tinfield-smoke-r2-20261013.ipynb'
kd=HERE/'kernel'; meta=json.loads((kd/'kernel-metadata.json').read_text()); nbp=kd/meta['code_file']
nb=json.loads(nbp.read_text()); base=json.loads(BASE.read_text()); r2=json.loads(R2.read_text())
nbformat.validate(nbformat.read(nbp,as_version=4)); shell=InteractiveShell.instance()
for i,c in enumerate(nb['cells']):
    if c['cell_type']=='code': compile(shell.transform_cell(''.join(c['source'])),f'cell-{i}','exec',flags=0x2000)
def bodies(n):
    out=[]
    for c in n['cells']:
        s=''.join(c['source'])
        if c['cell_type']!='code' or s.startswith('%%'): continue
        for node in ast.walk(ast.parse(shell.transform_cell(s))):
            if isinstance(node,ast.If) and isinstance(node.test,ast.Name) and node.test.id=='TRUE_SUBMISSION':
                out.append([ast.dump(x,include_attributes=False) for x in node.body])
    return out
assert bodies(nb)==bodies(base)
assert meta['title'].replace(' ','-')==meta['id'].split('/')[1]
chk=subprocess.run([sys.executable,'-B',str(REPO/'incumbent/check.py'),str(nbp),'--without','b1'],text=True,capture_output=True)
(HERE/'incumbent-check.log').write_text(chk.stdout+chk.stderr); assert chk.returncode==0, chk.stdout+chk.stderr
ex=subprocess.run([sys.executable,'-B',str(ROOT/'tinfield-r2/assembly_exercise.py'),str(nbp)],text=True,capture_output=True)
(HERE/'assembly-exercise.log').write_text(ex.stdout+ex.stderr); assert ex.returncode==0
receipt={'notebook':str(nbp.relative_to(REPO)),'base_sha256':hashlib.sha256(BASE.read_bytes()).hexdigest(),
 'notebook_sha256':hashlib.sha256(nbp.read_bytes()).hexdigest(),'metadata_sha256':hashlib.sha256((kd/'kernel-metadata.json').read_bytes()).hexdigest(),
 'changed_cells_vs_base':[i for i,(a,b) in enumerate(zip(base['cells'],nb['cells'])) if a['source']!=b['source']],
 'changed_cells_vs_r2':[i for i,(a,b) in enumerate(zip(r2['cells'],nb['cells'])) if a['source']!=b['source']],
 'submission_if_bodies_identical':True,'notebook_schema_valid':True,'all_code_cells_compile':True,'incumbent_check_without_b1':'OK',
 'assembly_exercise_real_indices':'OK (tinfield-r2/candidate-r3/assembly-exercise.log)','pushed':False}
(HERE/'LOCAL-QUALIFICATION.json').write_text(json.dumps(receipt,indent=2)+'\n'); print(json.dumps(receipt,indent=2))
