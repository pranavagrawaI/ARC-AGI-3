"""Explicit immutable preparation inputs; never operational receipts/authority."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def build():
 spec=json.loads((HERE/'packet-spec.json').read_text());names=set(spec['files'])|{'supervisor.py','http_once.py','build.py','packet.py','launch.py','reconciliation.py','prior-reconciliation-policy.json','transport.py','http_error_capture.py','bootstrap_patch.py','packet-spec.json','EXPERIMENT-CARD.md','READINESS-AUDIT.md','PRE-MORTEM.md','PREPARATION-PLAN.md','COLLECTION.md','SOURCE-BINDINGS.json','SOURCE-LINEAGE.json','INTEGRATION-DELTA.json','SUPERSESSION.json','verify.py','verify_bodies.py','verify_sdk.py','freeze_inputs.py','guard/sitecustomize.py','review-packet.json','SDK-PROOF.json'}
 for d in ('tests','collection_tools','review-packet','development-evidence','superseded-r1'):
  names|={str(p.relative_to(HERE)) for p in (HERE/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
 refs=dict(json.loads((HERE/'SOURCE-BINDINGS.json').read_text())['references'])
 policy=json.loads((HERE/'prior-reconciliation-policy.json').read_text());refs.update({str((HERE/n).resolve()):h for n,h in policy['files'].items()})
 return dict(status='UNASSIGNED_IMMUTABLE_REVIEW_INPUTS_NOT_AUTHORITY',files={n:sha(HERE/n) for n in sorted(names)},references=refs,excluded_mutable=['slot-confirmation.json','launch-attempt.json','launch-response.json','launch-error.json','quota-prelaunch.json','sessions-prelaunch.json','predecessor-status-prelaunch.json','failed-save-reconciliation-prelaunch.json','collection-space-prelaunch.json','kernel/','remote-readback/','output/','qualification/'])
if __name__=='__main__':
 with (HERE/'freeze.json').open('x') as f:json.dump(build(),f,indent=2);f.write('\n')
