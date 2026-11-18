#!/usr/bin/env python3
"""Write the immutable review manifest; mutable review/authorization/run receipts are excluded."""
import hashlib, json
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
PACKAGE=REPO/'deployment/kaggle'/HERE.name
ANALYSIS_FILES=['AUDIT-PLAN.md','EVIDENCE-ALLOWLIST.json','EXPERIMENT-CARD.md','R4-REVIEW-PREREQUISITE.json','SERVER-REFERENCE.json','accelerator.py','audit_run.py','case.json','collect_readonly.py','constructor.py','containment_base.py','inspection_runner.py','integration.py','integration_boundary.py','local_transport.py','make_manifest.py','model-metadata.json','prelaunch_check.py','PROVIDER-SELECTION.json','qualified_runner.py','real_trace.py','request_contract.py','tests/test_live_controls.py','tests/test_integration.py','tests/test_local_wire.py','tests/notebook_local_mock_runner.py']
PACKAGE_FILES=['BUILD.json','construction.ipynb','kernel-metadata.json','launch_once.py','reconcile_build.py','remote_entry.py']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    files={}
    for n in ANALYSIS_FILES: files['analysis/'+n]=sha(HERE/n)
    for n in PACKAGE_FILES: files['package/'+n]=sha(PACKAGE/n)
    m={'schema':'construction-policy-live-manifest-v1','identity':HERE.name,'reviewed_baseline':{'identity':'construction-policy-integration-20260913T221412Z-r4','manifest_sha256':'577c8d8b326d826c7e48e5b9a4e5cb668005eb4bd52c06e6d4ed4ed131fce451','notebook_sha256':'e4b9f30f8fbd84eefa3b1b84b2a00f19a379d9ee381b4a5e60d4141e794bfa35','build_sha256':'c4beed81a87f4ba667aedd0364c8132d7ea1c168ee7ad7c1cc06249f25df8f27'},'files':files,'mutable_exclusions':['AUTHORIZATION.json','REVIEW-RESOLVED.json','evidence/**','LAUNCH-CONSUMED.json','launch-receipt.json']}
    (HERE/'MANIFEST.json').write_text(json.dumps(m,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'manifest_sha256':sha(HERE/'MANIFEST.json'),'files':len(files)},indent=2))
if __name__=='__main__': main()
