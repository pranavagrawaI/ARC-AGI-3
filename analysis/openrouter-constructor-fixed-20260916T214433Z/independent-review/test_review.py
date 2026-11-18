import copy,json,sys,ast
from pathlib import Path
import pytest,jsonschema
B=Path(__file__).resolve().parents[1];OLD=B.parent/'openrouter-constructor-reasoning-20260916T185550Z'
sys.path[:0]=[str(B),str(B/'engine')]
from test_pair import Server,response,WORK,SOURCE
from run_pair import episode
from transport import initial_ledger
from constructor import _validate_work_item,work_item_schema

@pytest.mark.parametrize('arm,ordinal',[('off',3),('on',5)])
def test_previous_actual_bad_arguments_repair_at_real_boundary(tmp_path,arm,ordinal):
 raw=json.loads((OLD/'live'/arm/'child/raw'/f'{ordinal:03}.response.raw').read_text())
 args=json.loads(raw['choices'][0]['message']['tool_calls'][0]['function']['arguments'])
 assert args['work_item']==''
 ledger=tmp_path/'ledger.json';initial_ledger(ledger)
 with Server([response('execute_python',{'code':'print(1)'},1,arm),response('submit_candidate',args,2,arm),response('submit_candidate',{'work_item':WORK,'candidate_source':SOURCE.splitlines()},3,arm)]) as server:
  result=episode(arm,tmp_path/'arm',ledger,fixture_url=server.url)
 feedback=json.loads([m for m in server.requests[2]['messages'] if m['role']=='tool'][-1]['content'])
 assert 'arguments.work_item must be an object; received string' in feedback['error']
 assert 'not as JSON text or top-level argument fields' in feedback['error']
 assert result['frozen'] and result['repair']['accepted_after_repair']
 progress=json.loads((tmp_path/'arm/PROGRESS.json').read_text())
 assert [x['method'] for x in progress['evaluation']['rpc_methods']]==['initialize','predict','assimilate']
 captured=json.loads((tmp_path/'arm/raw/002.response.raw').read_text())
 assert json.loads(captured['choices'][0]['message']['tool_calls'][0]['function']['arguments'])==args

def test_inherited_acceptance_and_documented_schema_edge():
 tree=ast.parse((OLD/'engine/constructor.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_validate_work_item');scope={'Any':object};exec(compile(ast.Module(body=[fn],type_ignores=[]),'prior-validator','exec'),scope)
 old=scope['_validate_work_item'];schema=jsonschema.Draft202012Validator(work_item_schema(1))
 vals=[None,True,False,0,1,1.0,2,'',' ','1',[],[1],[True],[False],[1.0],['1'],{},'ground','needs_computation',['x'],[' ']]
 cases=[None,'',{},WORK]+[dict(WORK,**{k:v}) for k in WORK for v in vals]
 def accepts(fn,x):
  try:fn(copy.deepcopy(x),1);return True
  except Exception:return False
 mismatches=[]
 for c in cases:
  assert accepts(old,c)==accepts(_validate_work_item,c)
  if schema.is_valid(c)!=accepts(_validate_work_item,c):mismatches.append(c)
 assert len(mismatches)==2
 assert all(c['evidence_refs']==[1.0] or c['completion_checks']==[1.0] for c in mismatches)
