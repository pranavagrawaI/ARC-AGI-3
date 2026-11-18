"""Deterministic assembly only; no upload or launch."""
import argparse
import ast
import base64
import hashlib
import lzma
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
DONOR=HERE.parent/'actor-mtp-20260906'

def sha(data):return hashlib.sha256(data).hexdigest()

def build():
    raw=(DONOR/'actor_mtp.ipynb').read_bytes()
    donor=json.loads((DONOR/'manifest.json').read_text())
    assert sha(raw)==donor['notebook_sha256'],'Donor notebook drift'
    assert sha((DONOR/'runner.py').read_bytes())==donor['assets']['runner.py'],'Reference transport drift'
    pin=json.loads((HERE/'SOURCE-PIN.json').read_text())
    assert sha((HERE/'baseline.py').read_bytes())==pin['sha256']['runner.py'],'Baseline helper drift'
    notebook=json.loads(raw)
    bootstrap=ast.parse(''.join(notebook['cells'][3]['source']))
    install_calls=0
    for node in ast.walk(bootstrap):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='subprocess' and node.func.attr=='run':
            if not any(k.arg=='timeout' for k in node.keywords):
                node.keywords.append(ast.keyword(arg='timeout',value=ast.Constant(value=1200)))
            install_calls+=1
    assert install_calls==3,'Bootstrap command surface drift'
    notebook['cells'][3]['source']=ast.unparse(ast.fix_missing_locations(bootstrap))+'\n'
    # Kaggle rejects kernel source >=1 MiB. Plain base64 of these assets is
    # ~715 KB and overflows it; the captured corpus alone compresses ~25x.
    # Integrity is unchanged: the same SHA256 is asserted after decompression.
    assets=''.join(notebook['cells'][2]['source']).replace('import base64','import base64\nimport lzma',1)
    sources={'actor-replay.py':HERE/'runner.py','baseline.py':HERE/'baseline.py',
             'actor-reference.py':DONOR/'runner.py','prefix-corpus.json':HERE/'corpus.json',
             'watchdog.py':HERE.parent/'mtp-sampling-repair-20260906'/'watchdog.py',
             'prefix-assess.py':HERE/'assess.py'}
    hashes={}
    for name,path in sources.items():
        data=path.read_bytes();hashes[name]=sha(data)
        blob=base64.b64encode(lzma.compress(data,preset=9|lzma.PRESET_EXTREME)).decode()
        assets+=f"\n_data=lzma.decompress(base64.b64decode({blob!r}))\nassert hashlib.sha256(_data).hexdigest()=={sha(data)!r}\nPath('/tmp/{name}').write_bytes(_data)\n"
    notebook['cells'][2]['source']=assets
    notebook['cells'][0]['source']='# Chunk configuration and real successor prefix reuse\n2048/4096 ABBA; offline RTX PRO 6000. No gameplay, tools, submission or automatic promotion. Read EXPERIMENT-CARD.md.\n'
    notebook['cells'][4]['source']="""import subprocess, sys
from pathlib import Path
out=Path('/kaggle/working/prefix-cache-20260907');out.mkdir(exist_ok=True)
with (out/'runner.log').open('w') as log:
    result=subprocess.run([sys.executable,'/tmp/actor-replay.py'],stdout=log,stderr=subprocess.STDOUT,timeout=9300)
print((out/'runner.log').read_text()[-12000:])
assert result.returncode==0, 'Failed; preserve artifacts, do not automatically relaunch'
subprocess.run([sys.executable,'/tmp/prefix-assess.py',str(out)],check=True)
"""
    for cell in notebook['cells']:
        if cell['cell_type']=='code':compile(''.join(cell['source']),'notebook','exec')
    data=(json.dumps(notebook,indent=1)+'\n').encode()
    meta=json.loads((DONOR/'kernel-metadata.json').read_text())
    meta.update(id='pranavagrawai/prefix-cache-20260907',title='Prefix Cache Chunk ABBA 20260907',code_file='prefix_cache.ipynb')
    manifest={'schema':'prefix-cache-v1','notebook_sha256':sha(data),'assets':hashes,
        'donor_notebook_sha256':sha(raw),'source_pin_sha256':sha((HERE/'SOURCE-PIN.json').read_bytes()),
        'card_sha256':sha((HERE/'EXPERIMENT-CARD.md').read_bytes()),'chunks':[2048,4096,4096,2048],
        'request_cap':148,'runner_cap_s':9000,'outer_runner_cap_s':9300,'bootstrap_command_timeout_s':1200,'promotion':False,
        'upload_ready':True,'scope':'Prepared only; no launch/push authorization'}
    return {'prefix_cache.ipynb':data,'kernel-metadata.json':(json.dumps(meta,indent=2)+'\n').encode(),
            'manifest.json':(json.dumps(manifest,indent=2)+'\n').encode()}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args()
    for name,data in build().items():
        if args.check:assert (HERE/name).read_bytes()==data,name
        else:(HERE/name).write_bytes(data)
    print('Package verified; no upload or launch')
