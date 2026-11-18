"""Execute the generated notebook through its local HTTP mode."""
import json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE/'tests'))
from request_contract import LOCAL_MODEL
from test_integration import CANDIDATE,WORK
from test_local_wire import ScriptedServer,local_response

def tc(identifier,name,args):return {'id':identifier,'type':'function','function':{'name':name,'arguments':json.dumps(args)}}
responses=[
 local_response(tc('notebook.inspect/a','execute_python',{'code':"import json;print(len(json.load(open('/evidence/case.json'))['visible_transitions']))"})),
 local_response(tc('notebook.inspect/b','execute_python',{'code':"import json;print(json.load(open('/evidence/case.json'))['frontier'])"})),
 local_response(tc('notebook.submit/c','submit_candidate',{'work_item':WORK,'candidate_source':CANDIDATE})),
]
server=ScriptedServer(responses)
try:
 os.environ.pop('CONSTRUCTION_FIXTURE',None);os.environ['CONSTRUCTION_LOCAL_BASE_URL']=server.base
 notebook=Path(sys.argv[1]);code=json.loads(notebook.read_text())['cells'][1]['source']
 try:
  exec(compile(code,str(notebook),'exec'),{'__name__':'__main__'})
 except SystemExit as exit:
  if exit.code not in (None,0): raise
 Path('/kaggle/working/notebook-local-wire.json').write_text(json.dumps([raw.decode() for raw in server.requests],indent=2)+'\n')
finally:server.close()
