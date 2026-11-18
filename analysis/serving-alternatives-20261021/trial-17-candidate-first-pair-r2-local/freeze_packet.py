"""Freeze one reviewed local preparation snapshot; never arms/assigns/pushes."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent

def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def freeze():
 if Path('/proc/self/fd/1').resolve().is_relative_to(HERE):raise ValueError('freeze stdout receipt must be outside the hashed tree')
 for name in ('slot-confirmation.json','launch-attempt.json','launch-response.json','kernel','remote-readback'):
  if (HERE/name).exists():raise ValueError('unassigned local preparation only; unexpected operational state '+name)
 manifest=json.loads((HERE/'review-packet.json').read_bytes())
 if manifest['launch_authorized'] is not False:raise ValueError('review packet must be unassigned')
 original=HERE.parent/'trial-17-candidate-first-pair-local';old=json.loads((original/'freeze.json').read_bytes())
 refs={str((original/name).resolve()):digest for name,digest in old['files'].items()}|old['references']
 for name in ('freeze.json','HANDOFF.json','HANDOFF.md'):refs[str((original/name).resolve())]=sha(original/name)
 review=HERE.parent/'coordination-20261024'
 for name in ('trial17-independent-review-blocked.json','trial17-independent-qualification-root-r1','trial17-runtime-review','trial17-primary-deadline-review'):
  source=review/name
  for path in ([source] if source.is_file() else source.rglob('*')):
   if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc':refs[str(path.resolve())]=sha(path)
 basis=json.loads((HERE.parent/'candidate-first-paired-capacity-proposal-local/BASIS.json').read_bytes())
 refs.update(basis['sources_sha256'])
 for row in json.loads((HERE/'ORIGINS.json').read_bytes()).values():refs[row['path']]=row['sha256']
 bindings=json.loads((HERE/'qualification/final-exact/SDK-BINDINGS.json').read_bytes())['files'];refs.update(bindings)
 for path,digest in refs.items():
  if sha(path)!=digest:raise ValueError('preserved reference drift '+path)
 excluded={'freeze.json','HANDOFF.json','HANDOFF.md'}
 files={}
 for path in sorted(HERE.rglob('*')):
  if not path.is_file() or '__pycache__' in path.parts or path.suffix=='.pyc' or path.name in excluded:continue
  if path.is_symlink():raise ValueError('local preparation source/output symlink not permitted')
  files[str(path.relative_to(HERE))]=sha(path)
 value=dict(kind='LOCAL_EXECUTABLE_PREPARATION_UNASSIGNED',run_id=manifest['run_id'],files=files,references=refs,
  sources_sha256=manifest['sources_sha256'],config_sha256=manifest['config_sha256'],bindings=len(files)+len(refs),
  predecessor_bindings=21028,launch_authorized=False,API_authorized=False,GPU_authorized=False,slot_assigned=False,baseline65_rerun=False)
 with (HERE/'freeze.json').open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
 return dict(freeze_sha256=sha(HERE/'freeze.json'),local_files=len(files),reference_bindings=len(refs),bindings=value['bindings'])
if __name__=='__main__':print(json.dumps(freeze(),indent=2))
