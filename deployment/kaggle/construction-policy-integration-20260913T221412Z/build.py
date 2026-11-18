"""Build immutable Kaggle notebook from proven v15 bootstrap and integration assets."""
import base64,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
ANALYSIS=REPO/'analysis/construction-policy-integration-20260913T221412Z'
DONOR=HERE.parent/'submission-20260912-v15'
ASSETS=['accelerator.py','case.json','constructor.py','containment_base.py','integration.py','integration_boundary.py','model-metadata.json','PROVIDER-SELECTION.json','qualified_runner.py','real_trace.py','request_contract.py','remote_entry.py']
def build():
 donor=json.loads((DONOR/'submission.ipynb').read_text()); donor_meta=json.loads((DONOR/'kernel-metadata.json').read_text())
 encoded={n:base64.b64encode(((HERE/n) if n=='remote_entry.py' else (ANALYSIS/n)).read_bytes()).decode() for n in ASSETS}
 bootstrap="""import base64,hashlib,json,os,sys\nfrom pathlib import Path\nASSETS="""+repr(encoded)+"""\nEXPECTED="""+repr({n:hashlib.sha256(base64.b64decode(v)).hexdigest() for n,v in encoded.items()})+"""\nroot=Path('/tmp/construction-policy-integration'); root.mkdir(exist_ok=False)\nfor n,v in ASSETS.items():\n p=root/n;p.write_bytes(base64.b64decode(v))\n if hashlib.sha256(p.read_bytes()).hexdigest()!=EXPECTED[n]: raise RuntimeError('embedded asset drift:'+n)\nfor name in ('accelerator','constructor','containment_base','integration','integration_boundary','real_trace','request_contract'):\n sys.modules.pop(name,None)\nos.environ['CONSTRUCTION_ASSET_ROOT']=str(root)\n"""
 donor_assets=''.join(donor['cells'][1]['source'])
 remote="""\nif os.environ.get('CONSTRUCTION_FIXTURE')=='1':\n exec(compile((root/'remote_entry.py').read_text(),str(root/'remote_entry.py'),'exec'),{'__name__':'__main__'})\nelse:\n exec("""+repr(donor_assets)+""",globals(),globals())\n import subprocess\n command=[sys.executable,'/tmp/official_launcher_runtime.py','--output-dir','/kaggle/working/model-runtime','--manifest','/tmp/PROTOTYPE-qwen3_8_fp8-manifest.json','--input-root','/kaggle/input','--model-source','jakobbrggen/qwen3-8-27b-fp8-hf-snapshot/versions/1','--runtime-provenance','/tmp/sglang-runtime-provenance.json','--startup-timeout-s','1200','--request-timeout-s','180','--launcher-timeout-s','2700','--attention-backend','triton','--sampling-backend','pytorch','--profile-imports','--r8-l3b','--official-launcher-probe',str(root/'remote_entry.py'),'--launcher-python','/tmp/solver-venv/bin/python']\n subprocess.run(command,check=True,timeout=2900)\n"""
 notebook={'cells':[{'cell_type':'markdown','metadata':{},'source':'# Launch-guarded LS20 construction-policy integration\\nNo gameplay or competition submission.'},{'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':bootstrap+remote}],'metadata':{'kernelspec':{'name':'python3','language':'python','display_name':'Python 3'}},'nbformat':4,'nbformat_minor':5}
 meta={k:donor_meta[k] for k in ('language','kernel_type','is_private','enable_gpu','machine_shape','enable_internet','enable_tpu','dataset_sources','kernel_sources','competition_sources','model_sources')}
 meta.update(id='pranavagrawai/ls20-construction-policy-integration-20260913t221412z',title='LS20 Construction Policy Integration 20260913T221412Z',code_file='construction.ipynb',docker_image='gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461')
 (HERE/'construction.ipynb').write_text(json.dumps(notebook,indent=1)+'\n');(HERE/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
 manifest={'identity':'construction-policy-integration-20260913T221412Z','donor':str(DONOR.relative_to(REPO)),'donor_build_sha256':hashlib.sha256((DONOR/'BUILD.json').read_bytes()).hexdigest(),'assets':{n:hashlib.sha256(((HERE/n) if n=='remote_entry.py' else ANALYSIS/n).read_bytes()).hexdigest() for n in ASSETS},'files':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ('construction.ipynb','kernel-metadata.json')}}
 (HERE/'BUILD.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':build()
