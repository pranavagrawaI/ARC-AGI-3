import json,os,sys,time
from pathlib import Path
phase=sys.argv[1];root=Path(os.environ['SCREEN_OUTPUT']);base=root if root.name=='output' else root.parent
print('CPU_ONLY_STAGE '+phase,flush=True)
if phase=='pair.reduce':(base/'paired-result.json').write_text(json.dumps(dict(verdict='CLOSE_TESTED_CONFIG',ratio=1.0,CPU_ONLY_NOT_A_CAPACITY_RESULT=True)))
else:(base/(phase+'.cpu.json')).write_text(json.dumps(dict(phase=phase,identity={k:v for k,v in os.environ.items() if k.startswith('SCREEN_SERVER_')},CPU_ONLY=True)))
if phase==os.environ.get('CPU_FAIL_PHASE'):raise SystemExit(9)
