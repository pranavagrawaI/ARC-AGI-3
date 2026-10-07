"""One authorized retry through the original fail-closed single-attempt watcher."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from prepare import HERE, ORIGINAL, SLUG, TAG, COMPETITION

sys.path.insert(0, str(ORIGINAL))
import launch_watch
import verify_prelaunch


def verify_retry():
    parent = verify_prelaunch.verify()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--', '.'], cwd=HERE, text=True)
    if dirty:
        raise ValueError('retry differs from committed state: ' + dirty)
    commit = subprocess.check_output(['git', 'rev-parse', TAG + '^{commit}'], cwd=HERE, text=True).strip()
    if subprocess.check_output(['git', 'diff', commit, '--', '.'], cwd=HERE, text=True):
        raise ValueError('retry differs from frozen tag')
    manifest = json.loads((HERE / 'freeze.json').read_text())
    for name, digest in manifest['files'].items():
        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != digest:
            raise ValueError('retry hash mismatch: ' + name)
    metadata = json.loads((HERE / 'kernel/kernel-metadata.json').read_text())
    if metadata['competition_sources'] != [COMPETITION] or metadata['machine_shape'] != 'NvidiaRtxPro6000':
        raise ValueError('retry competition/accelerator mismatch')
    return {'run_id': manifest['run_id'], 'tag_commit': commit, 'parent': parent,
            'kernel_metadata': metadata, 'retry_files': manifest['files']}


if __name__ == '__main__':
    launch_watch.EVID = HERE / 'evidence'
    launch_watch.NOTEBOOK_DIR = HERE / 'kernel'
    launch_watch.identity = verify_retry
    # Force bounded requests even when the SDK explicitly passes timeout=None.
    import requests
    original_send = requests.Session.send
    def bounded_send(session, request, **kwargs):
        kwargs['timeout'] = (15, 45)
        return original_send(session, request, **kwargs)
    requests.Session.send = bounded_send
    launch_watch.main()
