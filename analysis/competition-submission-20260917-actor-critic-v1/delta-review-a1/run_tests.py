import os,json,subprocess,concurrent.futures
from pathlib import Path
O=Path(__file__).resolve().parent
commands=json.loads((O/'test-commands.json').read_text())
def run(c):
 env={k:v for k,v in os.environ.items() if k in ('HOME','PATH','LANG')};env.update(c['env']);env['TMPDIR']=str(O/'tmp');env['PYTHONDONTWRITEBYTECODE']='1'
 with (O/(c['name']+'.log')).open('w') as f: p=subprocess.run(c['argv'],env=env,stdout=f,stderr=subprocess.STDOUT)
 return dict(name=c['name'],returncode=p.returncode)
(O/'tmp').mkdir(exist_ok=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex: results=list(ex.map(run,commands))
(O/'test-results.json').write_text(json.dumps(results,indent=2));print(results)
