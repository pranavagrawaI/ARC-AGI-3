"""Explicit immutable input manifest. No live-root scan or operational output binding."""
import argparse,datetime as dt,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def create(final=False):
 assert sha(HERE/'supervisor.py')=='b48996d7a072d0f6ed058e04803d877c8d5ba37b505b04ffbb9e08ecb7f5d7f5','accepted supervisor changed'
 spec=json.loads((HERE/'packet-spec.json').read_text());report=json.loads((HERE/'review-packet.json').read_text())
 paths=set(spec['files'])|{'supervisor.py','http_once.py','packet-spec.json','review-packet.json','build.py','packet.py','transport.py','verify.py','launch.py','reconciliation.py','http_error_capture.py','prior-reconciliation-policy.json','EXPERIMENT-CARD.md','causal-model.json','api-source-review.json','control-source-extraction.json','freeze_review.py','DIAGNOSIS.md','PROPOSED-DELTA.json','LOCAL-QUALIFICATION.json','prove_health_local.py'}
 # These are immutable source/test/archive directories, never live operational output.
 for directory in ('source-review','tests','accepted-local-b48996d7','baseline'):
  paths.update(str(p.relative_to(HERE)) for p in (HERE/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
 paths.update('review-packet/'+n for n in (spec['metadata']['code_file'],'kernel-metadata.json'))
 for name in ('pinned-http_server.py','pinned-environ.py','pinned-tokenizer_manager.py','readiness-red.log','readiness-green.log','exception-preservation-red.log','exception-preservation-green.log','cleanup-race-red.log','cleanup-preservation-green.log','cleanup-context-green.log','ambient-exception-red.log','control-health-red.log','control-health-green.log','health-provenance-policy-green.log'):
  paths.add('evidence/'+name)
 policy=json.loads((HERE/'prior-reconciliation-policy.json').read_text());paths.update(policy['files'])
 prior=HERE.parent/'trial-8-wire-capacity-local';frozen=json.loads((prior/'freeze-r4.json').read_text());paths.add('../trial-8-wire-capacity-local/freeze-r4.json')
 for name,h in frozen['files'].items():
  assert sha(prior/name)==h,name
  paths.add('../trial-8-wire-capacity-local/'+name)
 if final:
  paths.update(('INTEGRATION-DELTA.json','HANDOFF.md','evidence/sdk-lossless.json','evidence/integrated-tests.log','evidence/LOCAL-INTEGRATION-QUALIFICATION.json','evidence/control-health-local-proof.json','evidence/integrated-tests-first.log','evidence/sdk-proof.log'))
 forbidden={'quota-prelaunch.json','sessions-prelaunch.json','predecessor-status-prelaunch.json','failed-save-reconciliation-prelaunch.json','slot-confirmation.json','launch-attempt.json','launch-response.json','launch-error.json'}
 assert not paths&forbidden
 assert not any(Path(n).parts[0] in ('kernel','output','runtime','remote-readback','collection') for n in paths)
 files={n:sha(HERE/n) for n in sorted(paths)}
 value=dict(schema='trial9-immutable-preparation-inputs-v1',status='LOCAL_REVIEW_READY_UNASSIGNED' if final else 'LOCAL_DRAFT_UNASSIGNED',created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),kernel_ref=spec['metadata']['id'],sources_sha256=report['sources_sha256'],config_sha256=report['config_sha256'],platform_hard_timeout_s=10200,cooperative_supervisor_timeout_s=9900,launch_authorized=False,predecessor_freeze_sha256=sha(prior/'freeze-r4.json'),excluded_operational_outputs=sorted(forbidden),files=files)
 (HERE/'freeze.json').write_text(json.dumps(value,indent=2)+'\n')
 receipt={k:value[k] for k in ('status','kernel_ref','sources_sha256','config_sha256','platform_hard_timeout_s','cooperative_supervisor_timeout_s','launch_authorized')};receipt.update(freeze_sha256=sha(HERE/'freeze.json'),bindings=len(files),reconciliation_policy_sha256=sha(HERE/'prior-reconciliation-policy.json'))
 (HERE/'FREEZE-RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');create(p.parse_args().final)
