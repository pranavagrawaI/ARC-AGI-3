import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1];PACKAGE=REPO/'deployment/kaggle'/HERE.name
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def make():
 analysis_files=[p for p in HERE.rglob('*') if p.is_file() and 'evidence' not in p.parts and '__pycache__' not in p.parts and p.name not in ('MANIFEST.json','READY.md')]
 package_files=[p for p in PACKAGE.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
 value={'identity':HERE.name,'analysis':{str(p.relative_to(HERE)):digest(p) for p in sorted(analysis_files)},'package':{str(p.relative_to(PACKAGE)):digest(p) for p in sorted(package_files)},'prerequisites':{'remote_containment_result_sha256':'bccb2dddb20b70a8a25ffb6cfccbbf2a63f20468815ba87d6c2207c6b314ffe7','remote_containment_final_manifest_sha256':'4e674111eb10784ad4086c589ed4f615a7cb067cb27da66f44dcc9b985fd696a','v15_metadata_sha256':'46d0b95ad2cc8bcac7c0c0c49cad4f910a0daf87e623ec764c1d325b7404958a'}}
 (HERE/'MANIFEST.json').write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':make()
