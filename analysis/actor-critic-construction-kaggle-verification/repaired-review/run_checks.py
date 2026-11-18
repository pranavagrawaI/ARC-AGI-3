"""Read-only frozen-candidate checks; all persistent review output stays here."""
import json, os, pathlib, subprocess, sys
R = pathlib.Path('/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus')
OUT = pathlib.Path(__file__).resolve().parent
PY = R/'deployment/kaggle/.venv/bin/python'
A = R/'analysis/actor-critic-construction-kaggle-launch'
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(OUT/'tmp'))
(OUT/'tmp').mkdir(exist_ok=True)
results = []
for name, command, timeout in [('verify-before', [str(PY), '-B', str(A/'verify.py')], 60), ('suite', ['unshare','--user','--map-auto','--map-root-user',str(PY),'-B','-m','pytest','-v','-p','no:cacheprovider', '--basetemp',str(OUT/'pytest-work'),str(A/'tests')], 300), ('verify-after',[str(PY),'-B',str(A/'verify.py')],60)]:
    p = subprocess.run(command, cwd=R, env=env, capture_output=True, text=True, timeout=timeout)
    (OUT/(name+'.stdout')).write_text(p.stdout)
    (OUT/(name+'.stderr')).write_text(p.stderr)
    results.append({'check':name,'returncode':p.returncode,'command':command})
(OUT/'checks.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
