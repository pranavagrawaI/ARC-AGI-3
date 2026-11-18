import sys, json, importlib.util, socket
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, '/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/analysis/r10-prep-20260916/local-official-loop/test_actual_supervisor_worker_0/actor-critic-construction')
def network(event, args):
    if event == 'socket.connect' and args[1][0] not in ('127.0.0.1', '::1'):
        raise RuntimeError('fixture forbids non-loopback network')
sys.addaudithook(network)
import gameplay_play
gameplay_play.inspect_gpu = lambda out: (out / 'GPU-FIXTURE.json').write_text(json.dumps({'evidence_kind': 'synthetic hardware context, NOT GPU observation'}))
gameplay_play.ADMISSION_SECONDS = 30
gameplay_play.MAX_ACTIONS = 3
gameplay_play.ARM_CALL_CAP = {'A': 13, 'B': 13}
sys.path.insert(0, '/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/analysis/r10-prep-20260916/tests')
from game_mount import enter
enter(Path('/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/analysis/r10-prep-20260916/local-official-loop/test_actual_supervisor_worker_0/default-game-mount'))
assert '--environments' not in sys.argv
import os
os.environ['OPERATION_MODE'] = 'ONLINE'  # explicit OFFLINE must win
sys.argv += ["--working", '/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/analysis/r10-prep-20260916/local-official-loop/test_actual_supervisor_worker_0/working']
raise SystemExit(gameplay_play.main())
