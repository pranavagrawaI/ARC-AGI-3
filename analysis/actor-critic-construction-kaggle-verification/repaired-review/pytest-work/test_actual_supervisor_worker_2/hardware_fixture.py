import sys, json, importlib.util, socket
from pathlib import Path
sys.path.insert(0, '/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/analysis/actor-critic-construction-kaggle-verification/repaired-review/pytest-work/test_actual_supervisor_worker_2/actor-critic-construction')
def network(event, args):
    if event == 'socket.connect' and args[1][0] not in ('127.0.0.1', '::1'):
        raise RuntimeError('fixture forbids non-loopback network')
sys.addaudithook(network)
import kaggle_probe
kaggle_probe.inspect_gpu = lambda out: (out / 'GPU-FIXTURE.json').write_text(json.dumps({'evidence_kind': 'synthetic hardware context, NOT GPU observation'}))
sys.argv += ['--environments', '/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/environment_files']
raise SystemExit(kaggle_probe.main())
