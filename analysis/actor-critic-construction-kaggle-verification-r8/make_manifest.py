"""Coordinator-owned run-dir manifest/authorization writer. Read-only over sources; never pushes."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('gate', HERE / 'launch_control.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
RUN = gate.RUN_DIR
CANDIDATE = ROOT / 'analysis/actor-critic-construction-kaggle-r8-gameplay-candidate'


def main():
    for name in ('manifest.json', 'authorization.json', 'launch-review.json', 'CONSUMED.json', 'launch.json'):
        assert not (RUN / name).exists(), f'{name} already exists; identity is single-use'
    files = {gate.SCRIPT, gate.SCRIPT.with_name('test_launch_control.py'), RUN / 'EXPERIMENT-CARD.md', gate.LOCAL_REVIEW}
    files |= {p for p in gate.PACKAGE_DIR.rglob('*') if p.is_file()}
    files |= {p for p in CANDIDATE.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    sources = {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}
    card = sha(RUN / 'EXPERIMENT-CARD.md')
    manifest = dict(schema=1, identity=gate.IDENTITY, run_dir=str(RUN), package_dir=str(gate.PACKAGE_DIR),
                    target=gate.TARGET, card_sha256=card, notebook_sha256=gate.NOTEBOOK_HASH,
                    policy=dict(schema=1, identity=gate.IDENTITY, target=gate.TARGET, limits=gate.LIMITS, private=True,
                                internet=False, accelerator='NvidiaRtxPro6000', competition_submission=False,
                                dataset_push=False, local_monitor_cancels_remote=False),
                    sources=sources)
    gate.exclusive(RUN / 'manifest.json', manifest)
    digest = sha(RUN / 'manifest.json')
    gate.exclusive(RUN / 'authorization.json', dict(schema=1, identity=gate.IDENTITY, target=gate.TARGET,
        manifest_sha256=digest, card_sha256=card, notebook_sha256=gate.NOTEBOOK_HASH,
        max_pushes=1, retries=0, timeout_seconds=3600, used=False))
    print(json.dumps(dict(manifest_sha256=digest, card_sha256=card, sources=len(sources)), indent=2))


if __name__ == '__main__':
    sys.exit(main())
