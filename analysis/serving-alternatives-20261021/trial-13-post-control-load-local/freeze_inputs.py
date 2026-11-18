"""Freeze explicit immutable packet inputs, never live operational receipts."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def build():
    spec=json.loads((HERE/'packet-spec.json').read_text())
    names=set(spec['files'])|{'supervisor.py','http_once.py','build.py','packet.py','launch.py','reconciliation.py','prior-reconciliation-policy.json','transport.py','http_error_capture.py','packet-spec.json','SOURCE-LINEAGE.json','EXPERIMENT-CARD.md','PRE-MORTEM.md','COLLECTION.md','PREPARATION-PLAN.md','verify_sdk.py','freeze_inputs.py','verify.py','verify_runtime.py','bootstrap_patch.py','review-packet.json','SDK-PROOF.json','guard/sitecustomize.py'}
    for directory in ('tests','collection_tools'):
        names|={str(p.relative_to(HERE)) for p in (HERE/directory).glob('*.py')}
    names|={str(p.relative_to(HERE)) for p in (HERE/'review-packet').iterdir() if p.is_file()}
    # The single failed baseline and raw case evidence are immutable review input.
    names|={str(p.relative_to(HERE)) for p in (HERE/'development-evidence/runtime-first').rglob('*') if p.is_file()}
    refs=dict(json.loads((HERE/'SOURCE-LINEAGE.json').read_text())['references'])
    policy=json.loads((HERE/'prior-reconciliation-policy.json').read_text());refs.update({str((HERE/name).resolve()):expected for name,expected in policy['files'].items()})
    return dict(status='UNASSIGNED_REVIEW_INPUTS_NOT_LAUNCH_AUTHORITY',files={name:sha(HERE/name) for name in sorted(names)},references=refs,
        excluded_mutable=['slot-confirmation.json','launch-attempt.json','launch-response.json','launch-error.json','quota-prelaunch.json','sessions-prelaunch.json','predecessor-status-prelaunch.json','failed-save-reconciliation-prelaunch.json','kernel/','output/','remote-readback/','qualification/'])
if __name__=='__main__':
    with (HERE/'freeze.json').open('x') as stream:json.dump(build(),stream,indent=2);stream.write('\n')
