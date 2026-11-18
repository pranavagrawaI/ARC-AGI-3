"""Freeze allowlisted authored sources; review/authorization are deliberately external."""
from pathlib import Path
import json,hashlib
RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[1];P=ROOT/'deployment/kaggle/submission-20260917-actor-critic-competition-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if __name__=='__main__':
    files=[p for p in P.rglob('*') if p.is_file()]
    files += [RUN/n for n in ('EXPERIMENT-CARD.md','launch_control.py','test_gate.py','freeze.py','qualify_gateway.py','test_local.py','exact_version.py','test_entrypoints.py','test_deadlines.py','test_budget_accounting.py','test_adapter.py','finalize.py','finalize_a1.py')]
    import launch_control as gate
    m=dict(schema=1,identity=gate.IDENTITY,target=gate.TARGET,run_dir=str(RUN),package_dir=str(P),command=gate.COMMAND,metadata=json.loads((P/'kernel-metadata.json').read_text()),card_sha256=sha(RUN/'EXPERIMENT-CARD.md'),notebook_sha256=sha(P/'submission.ipynb'),sources={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)})
    (RUN/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    print(sha(RUN/'manifest.json'))
