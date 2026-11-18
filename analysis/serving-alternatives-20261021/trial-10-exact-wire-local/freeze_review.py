"""Explicit immutable preparation boundary, with flattened audit/dependency refs."""
import argparse,datetime as dt,hashlib,json,os
from pathlib import Path
import audit_bindings
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def create(final=False):
 spec=json.loads((HERE/'packet-spec.json').read_text());report=json.loads((HERE/'review-packet.json').read_text())
 names=set(spec['files'])|{'supervisor.py','http_once.py','packet-spec.json','review-packet.json','build.py','packet.py','transport.py','verify.py','launch.py','reconciliation.py','http_error_capture.py','prior-reconciliation-policy.json','EXPERIMENT-CARD.md','causal-model.json','api-source-review.json','control-source-extraction.json','freeze_review.py','audit_bindings.py','audit-bindings.json','pin_dependencies.py','prove_health_local.py','prove_bodies.py'}
 for directory in ('source-review','tests','accepted-local-b48996d7','baseline','evidence'):
  names.update(str(p.relative_to(HERE)) for p in (HERE/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
 names.update('review-packet/'+n for n in (spec['metadata']['code_file'],'kernel-metadata.json'))
 names.update(os.path.relpath(p,HERE) for p in audit_bindings.paths())
 policy=json.loads((HERE/'prior-reconciliation-policy.json').read_text());names.update(policy['files'])
 prior=HERE.parent/'trial-9-readiness-local';priorfreeze=json.loads((prior/'freeze.json').read_text());names.add('../trial-9-readiness-local/freeze.json')
 for name,digest in priorfreeze['files'].items():
  assert sha(prior/name)==digest,name
  names.add(os.path.relpath((prior/name).resolve(),HERE))
 if final:names.update(('INTEGRATION-DELTA.json','HANDOFF.md','LOCAL-QUALIFICATION.json'))
 forbidden={'quota-prelaunch.json','sessions-prelaunch.json','predecessor-status-prelaunch.json','failed-save-reconciliation-prelaunch.json','slot-confirmation.json','launch-attempt.json','launch-response.json','launch-error.json'}
 assert not names&forbidden
 assert not any(Path(n).parts[0] in ('kernel','output','runtime','remote-readback','collection') for n in names)
 files={name:sha(HERE/name) for name in sorted(names)}
 value=dict(schema='trial10-immutable-preparation-inputs-v1',status='LOCAL_REVIEW_READY_UNASSIGNED' if final else 'LOCAL_DRAFT_UNASSIGNED',created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),kernel_ref=spec['metadata']['id'],sources_sha256=report['sources_sha256'],config_sha256=report['config_sha256'],platform_hard_timeout_s=10200,cooperative_supervisor_timeout_s=9900,launch_authorized=False,predecessor_freeze_sha256=sha(prior/'freeze.json'),excluded_operational_outputs=sorted(forbidden),files=files)
 (HERE/'freeze.json').write_text(json.dumps(value,indent=2)+'\n')
 receipt={k:value[k] for k in ('status','kernel_ref','sources_sha256','config_sha256','platform_hard_timeout_s','cooperative_supervisor_timeout_s','launch_authorized')};receipt.update(freeze_sha256=sha(HERE/'freeze.json'),bindings=len(files),reconciliation_policy_sha256=sha(HERE/'prior-reconciliation-policy.json'))
 (HERE/'FREEZE-RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');create(p.parse_args().final)
