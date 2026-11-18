import sys,runpy,tempfile
sys.path.insert(0,'/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/analysis/actor-critic-construction-kaggle-verification/repaired-review/cold-entry/actor-critic-construction')
tempfile.tempdir='/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/analysis/actor-critic-construction-kaggle-verification/repaired-review/tmp'
def offline(event,args):
    if event in {'socket.connect','socket.bind','socket.getaddrinfo'}: raise RuntimeError('review forbids network')
sys.addaudithook(offline)
import accelerator
# Supply only a synthetic executable path, preserving real GPU query/parsing.
accelerator.discover.__defaults__=(['/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/analysis/actor-critic-construction-kaggle-verification/repaired-review/cold-entry/synthetic-nvidia-smi'],)
runpy.run_path('/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/analysis/actor-critic-construction-kaggle-verification/repaired-review/cold-entry/actor-critic-construction/donor_bootstrap.py',run_name='__main__')
