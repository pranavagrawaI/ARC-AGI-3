"""Review-gated one-shot R10 freeze; local files only, never provider admission."""
import argparse
import json
from pathlib import Path
import sys
import launch_control as gate


def freeze(run_dir):
    run=gate.exact_run(run_dir)
    for name in ('manifest.json','authorization.json','launch-review.json','hermes-release.json','CONSUMED.json','launch.json'):
        gate.require(not (run/name).exists() and not (run/name).is_symlink(), f'{name} already exists; no re-freeze')
    gate.validate_package()  # Requires exact affirmative independent package review BEFORE any writes.
    gate.require((run/'EXPERIMENT-CARD.md').read_bytes()==(gate.PACKAGE_DIR/'EXPERIMENT-CARD.md').read_bytes(), 'card copies differ')
    metadata=gate.load(gate.PACKAGE_DIR/'kernel-metadata.json')
    gate.matches(metadata,dict(id=gate.TARGET,is_private=True,enable_internet=False,competition_sources=['arc-prize-2026-arc-agi-3']))
    files={gate.SCRIPT,gate.SCRIPT.with_name('test_launch_control.py'),gate.SCRIPT.with_name('make_manifest.py'),gate.SCRIPT.with_name('test_make_manifest.py'),gate.INVENTORY,gate.LOCAL_REVIEW,run/'EXPERIMENT-CARD.md'}
    files.update(p for p in gate.PACKAGE_DIR.rglob('*') if p.is_file())
    sources={}
    for p in sorted(files):
        gate.require(p.resolve()==p and p.is_file() and p.is_relative_to(gate.ROOT), 'unsafe freeze source')
        sources[str(p.relative_to(gate.ROOT))]=gate.sha(p)
    # Recheck review/package immediately before committing the single-use manifest.
    gate.validate_package()
    gate.require(all(gate.sha(gate.ROOT/n)==h for n,h in sources.items()),'freeze source changed')
    card=gate.sha(run/'EXPERIMENT-CARD.md')
    manifest=dict(schema=1,identity=gate.IDENTITY,run_dir=str(run),package_dir=str(gate.PACKAGE_DIR),target=gate.TARGET,card_sha256=card,notebook_sha256=gate.NOTEBOOK_HASH,sources=sources,
        policy=dict(schema=1,identity=gate.IDENTITY,target=gate.TARGET,limits=gate.LIMITS,private=True,internet=False,accelerator='NvidiaRtxPro6000',competition_submission=False,dataset_push=False,local_monitor_cancels_remote=False))
    gate.exclusive(run/'manifest.json',manifest)
    digest=gate.sha(run/'manifest.json')
    gate.exclusive(run/'authorization.json',dict(schema=1,identity=gate.IDENTITY,target=gate.TARGET,manifest_sha256=digest,card_sha256=card,notebook_sha256=gate.NOTEBOOK_HASH,max_pushes=1,retries=0,timeout_seconds=3600,used=False))
    return dict(status='FROZEN_PENDING_LAUNCH_REVIEW_AND_HERMES_RELEASE',manifest_sha256=digest,sources=len(sources),provider_calls=0)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True);a=p.parse_args(argv)
    try:print(json.dumps(freeze(a.run_dir),sort_keys=True));return 0
    except (OSError,ValueError,TypeError,KeyError) as error:
        print(json.dumps({'status':'HOLD','reason':str(error),'provider_calls':0}));return 2
if __name__=='__main__':sys.exit(main())
