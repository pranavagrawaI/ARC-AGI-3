"""Prepare two private public-screen notebook artifacts; never uploads or launches."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
parent=ROOT/'solvers/m2_f1x_reap/experiment/screen-1/merge/kernel'
p=parent/'m2-f1x-reap14-screen1.ipynb';raw=p.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='9d634d6accb024d53b875106c4d77ad12948d895ea485cd4be44b7b4f83f56c0'
base=json.loads(raw);meta=json.loads((parent/'kernel-metadata.json').read_text());manifest={'parent_sha256':hashlib.sha256(raw).hexdigest(),'purpose':'Local preparation only; fresh experiment authority and operational preflight required before any upload or launch.','arms':{}}
needle="    'ARC3_ANIMATION': '1',\n";replacement=needle+"    'ARC3_ANIMATION_IMAGE': 'peak',\n"
assert ''.join(base['cells'][5]['source']).count(needle)==1
for arm in ['control','peak']:
 dest=OUT/'prepared'/arm;dest.mkdir(parents=True,exist_ok=True)
 nb=json.loads(raw)
 if arm=='peak':nb['cells'][5]['source']=''.join(nb['cells'][5]['source']).replace(needle,replacement).splitlines(keepends=True)
 expected=[5] if arm=='peak' else []
 assert [i for i,(a,b) in enumerate(zip(base['cells'],nb['cells'])) if a!=b]==expected
 if arm=='peak':
  reverse=json.loads(json.dumps(nb));reverse['cells'][5]['source']=''.join(reverse['cells'][5]['source']).replace(replacement,needle).splitlines(keepends=True);assert reverse==base
 m=json.loads(json.dumps(meta));slug=f'm2-animation-{arm}-screen1-20261027';name=slug+'.ipynb';m.update(id='pranavagrawai/'+slug,title=f'M2 automatic animation {arm} screen1 20261027',code_file=name)
 content=raw if arm=='control' else (json.dumps(nb,indent=1)+'\n').encode()
 (dest/name).write_bytes(content);(dest/'kernel-metadata.json').write_text(json.dumps(m,indent=2)+'\n')
 manifest['arms'][arm]={'notebook':str((dest/name).relative_to(ROOT)),'sha256':hashlib.sha256(content).hexdigest(),'metadata_sha256':hashlib.sha256((dest/'kernel-metadata.json').read_bytes()).hexdigest(),'changed_cells':expected}
(OUT/'PREPARATION.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
