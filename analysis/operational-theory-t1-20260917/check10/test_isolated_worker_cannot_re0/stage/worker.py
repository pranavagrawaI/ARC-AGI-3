import json,os,socket
from pathlib import Path
host = Path('/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/analysis/operational-theory-t1-20260917/check10/test_isolated_worker_cannot_re0/sentinel')
assert not host.exists()
assert not Path('/package/evidence').exists()
assert 'OPENROUTER_API_KEY' not in os.environ
try:
 socket.create_connection(('1.1.1.1',443),timeout=.1)
except OSError: pass
else: raise AssertionError('network escape')
try:
 Path('/package/injected.py').write_text('escape')
except OSError: pass
else: raise AssertionError('read-only package escape')
print('isolated')
