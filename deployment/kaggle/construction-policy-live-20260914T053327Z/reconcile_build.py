#!/usr/bin/env python3
"""Reconcile the live package without rebuilding executable bytes."""
import hashlib, json
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
ANALYSIS=REPO/'analysis'/HERE.name
R4=HERE.parent/'construction-policy-integration-20260913T221412Z-r4'
R4A=REPO/'analysis/construction-policy-integration-20260913T221412Z-r4'
ASSETS=['PROVIDER-SELECTION.json','accelerator.py','case.json','constructor.py','containment_base.py','inspection_runner.py','integration.py','integration_boundary.py','local_transport.py','model-metadata.json','qualified_runner.py','real_trace.py','request_contract.py','remote_entry.py']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main(write=True):
    assert sha(HERE/'construction.ipynb')==sha(R4/'construction.ipynb')=='e4b9f30f8fbd84eefa3b1b84b2a00f19a379d9ee381b4a5e60d4141e794bfa35'
    assert sha(HERE/'remote_entry.py')==sha(R4/'remote_entry.py')
    for n in ASSETS:
        if n!='remote_entry.py': assert sha(ANALYSIS/n)==sha(R4A/n), n
    expected=json.loads((ANALYSIS/'SERVER-REFERENCE.json').read_text())
    assert json.loads((HERE/'kernel-metadata.json').read_text())==expected
    build={'identity':HERE.name,'baseline_identity':R4.name,'baseline_manifest_sha256':'577c8d8b326d826c7e48e5b9a4e5cb668005eb4bd52c06e6d4ed4ed131fce451','notebook_inherited_byte_identical':True,'assets':{n:sha(HERE/n if n=='remote_entry.py' else ANALYSIS/n) for n in ASSETS},'files':{'construction.ipynb':sha(HERE/'construction.ipynb'),'kernel-metadata.json':sha(HERE/'kernel-metadata.json')}}
    if write: (HERE/'BUILD.json').write_text(json.dumps(build,indent=2,sort_keys=True)+'\n')
    return build
if __name__=='__main__': print(json.dumps(main(),indent=2,sort_keys=True))
