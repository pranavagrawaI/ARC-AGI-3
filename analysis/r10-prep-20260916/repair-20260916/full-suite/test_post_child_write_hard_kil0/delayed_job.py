import sys,time
sys.path.insert(0,'/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/analysis/r10-prep-20260916/r10a-package')
from persist_job import main
rc=main()
assert rc==0
time.sleep(30)
