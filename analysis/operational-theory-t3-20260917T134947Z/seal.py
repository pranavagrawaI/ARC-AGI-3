import hashlib,json,os,stat
from pathlib import Path
from dotenv import dotenv_values
out=Path(Path('/tmp/arc-t3-operator/output-path').read_text());root=Path.cwd()
secret=os.environ.get('OPENROUTER_API_KEY') or dotenv_values(root/'.env').get('OPENROUTER_API_KEY')
assert secret
files={};links={}
for p in sorted(out.rglob('*')):
 if p.is_symlink():links[str(p.relative_to(out))]=os.readlink(p)
 elif p.is_file():
  raw=p.read_bytes()
  assert secret.encode() not in raw,'credential material detected; artifact not sealed'
  files[str(p.relative_to(out))]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
del secret
claim=root/'analysis/operational-theory-t3-claims/992cc66ae2e8577f69f852f00a5330f8f0fb2269155c435f4d1d35ba2931c53c.json'
manifest={'task':'T3','status':'COMPLETE','source_manifest_sha256':'992cc66ae2e8577f69f852f00a5330f8f0fb2269155c435f4d1d35ba2931c53c','files':files,'symlinks':links,'one_shot_claim_sha256':hashlib.sha256(claim.read_bytes()).hexdigest(),'credential_scan':'no loaded credential bytes present in artifacts','provider_retries':0,'self_excluded':'ARTIFACT-MANIFEST.json'}
target=out/'ARTIFACT-MANIFEST.json'
with target.open('x') as f:f.write(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
for name,entry in files.items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==entry['sha256']
for p in out.rglob('*'):
 if p.is_symlink():continue
 if p.is_file():p.chmod(stat.S_IMODE(p.stat().st_mode)&~0o222)
for p in sorted((p for p in out.rglob('*') if p.is_dir() and not p.is_symlink()),key=lambda p:len(p.parts),reverse=True):p.chmod(stat.S_IMODE(p.stat().st_mode)&~0o222)
out.chmod(stat.S_IMODE(out.stat().st_mode)&~0o222);claim.chmod(0o444)
print(json.dumps({'launch_dir':str(out),'artifact_manifest_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'authorization_sha256':files['T3-AUTHORIZATION.json']['sha256'],'result_sha256':files['RESULT.json']['sha256'],'files':len(files),'sealed_read_only':True},indent=2))
