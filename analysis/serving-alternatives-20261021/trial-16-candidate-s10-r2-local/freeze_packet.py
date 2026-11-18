"""Bind this UNASSIGNED local review packet and preserved predecessor inputs."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    prior=HERE.parent/'trial-16-candidate-s10-local';old=json.loads((prior/'freeze.json').read_text());refs={}
    for name,expected in old['files'].items():refs[str((prior/name).resolve())]=expected
    for name,expected in old['references'].items():refs[str(Path(name).resolve())]=expected
    policy=json.loads((HERE/'prior-reconciliation-policy.json').read_text())
    for name,expected in policy['files'].items():refs[str((HERE/name).resolve())]=expected
    refs[str(prior/'freeze.json')]=sha(prior/'freeze.json')
    # Focused tests import retained fixtures from10; bind the actual sources.
    for path in (HERE.parent/'trial-10-exact-wire-local/tests').glob('*.py'):refs[str(path.resolve())]=sha(path)
    for name,expected in refs.items():
        if sha(name)!=expected:raise ValueError('preserved input changed '+name)
    forbidden={'freeze.json','HANDOFF.json','HANDOFF.md','review-summary.html','freeze-build.log'}
    files={str(path.relative_to(HERE)):sha(path) for path in sorted(HERE.rglob('*'))
        if path.is_file() and not path.is_symlink() and '__pycache__' not in path.parts
        and path.suffix!='.pyc' and path.name not in forbidden and not path.name.startswith('HANDOFF')}
    if any(name.startswith(('kernel/','remote-readback/')) or name in ('slot-confirmation.json','launch-attempt.json','launch-response.json') for name in files):raise ValueError('live operational state in review freeze')
    manifest=json.loads((HERE/'review-packet.json').read_text())
    value=dict(kind='UNASSIGNED_CANDIDATE_ALONE_S10_REVIEW',run_id=manifest['run_id'],files=files,references=refs,
        sources_sha256=manifest['sources_sha256'],config_sha256=manifest['config_sha256'],
        bindings=len(files)+len(refs),launch_authorized=False,baseline65_rerun=False)
    with (HERE/'freeze.json').open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(freeze_sha256=sha(HERE/'freeze.json'),files=len(files),references=len(refs),bindings=value['bindings'])))
if __name__=='__main__':main()
