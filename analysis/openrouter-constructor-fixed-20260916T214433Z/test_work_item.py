"""Regressions for paid-run interface failure, via HTTP + constructor + sealed ABI."""
import copy,json,sys
from pathlib import Path
import pytest
import jsonschema
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE/'engine')]
from test_pair import Server,response,WORK,SOURCE
from run_pair import episode
from transport import initial_ledger

CASES=[
 ('string',{'work_item':json.dumps(WORK)},'arguments.work_item must be an object; received string'),
 ('empty',{'work_item':{}},'missing keys'),
 ('flattened',dict(WORK),'arguments.work_item is missing'),
 ('double_nested',{'work_item':{'work_item':WORK}},'extra keys'),
 ('missing',{'work_item':{k:v for k,v in WORK.items() if k!='claim'}},'missing keys: claim'),
 ('extra',{'work_item':dict(WORK,extra='x')},'extra keys: extra'),
 ('null',{'work_item':None},'received null'),
 ('array',{'work_item':[]},'received array'),
 ('text_type',{'work_item':dict(WORK,decision=3)},'work_item.decision must be a string'),
 ('empty_text',{'work_item':dict(WORK,claim='  ')},'work_item.claim must contain non-whitespace text'),
 ('operation_type',{'work_item':dict(WORK,operation=[])},'work_item.operation must be a string'),
 ('operation_enum',{'work_item':dict(WORK,operation='invent')},'work_item.operation must be one of'),
 ('reasoning_enum',{'work_item':dict(WORK,reasoning_need='invent')},'work_item.reasoning_need must be one of'),
 ('refs_type',{'work_item':dict(WORK,evidence_refs='1')},'work_item.evidence_refs must be a nonempty array'),
 ('refs_empty',{'work_item':dict(WORK,evidence_refs=[])},'work_item.evidence_refs must be a nonempty array'),
 ('refs_item',{'work_item':dict(WORK,evidence_refs=['1'])},'work_item.evidence_refs[0] must be an integer'),
 ('refs_range',{'work_item':dict(WORK,completion_checks=[2])},'work_item.completion_checks[0] must cite a visible step in 1..1'),
 ('deps_type',{'work_item':dict(WORK,dependencies={})},'work_item.dependencies must be a nonempty array'),
 ('deps_item',{'work_item':dict(WORK,dependencies=[3])},'work_item.dependencies[0] must be a string'),
 ('deps_blank',{'work_item':dict(WORK,dependencies=[' '])},'work_item.dependencies[0] must contain non-whitespace text'),
]

@pytest.mark.parametrize('name,arguments,expected',CASES,ids=[c[0] for c in CASES])
def test_invalid_nested_work_item_feedback_then_valid_object_reaches_real_abi(tmp_path,name,arguments,expected):
 ledger=tmp_path/'ledger.json';initial_ledger(ledger)
 bad=dict(arguments,candidate_source=SOURCE.splitlines())
 replies=[response('execute_python',{'code':'import json\nprint(json.load(open("/evidence/case.json"))["frontier"])'},1),response('submit_candidate',bad,2),response('submit_candidate',{'work_item':WORK,'candidate_source':SOURCE.splitlines()},3)]
 with Server(replies) as server:r=episode('off',tmp_path/'arm',ledger,fixture_url=server.url)
 feedback=json.loads([m for m in server.requests[2]['messages'] if m['role']=='tool'][-1]['content'])
 assert feedback['status']=='rejected' and expected in feedback['error']
 assert r['frozen'] and r['submissions']==2 and r['repair']['accepted_after_repair']
 progress=json.loads((tmp_path/'arm/PROGRESS.json').read_text())
 assert [x['method'] for x in progress['evaluation']['rpc_methods']]==['initialize','predict','assimilate']
 assert progress['evaluation']['visible']['status']=='passed'
 # Original malformed argument remains in raw receipt: no implicit coercion.
 raw=json.loads((tmp_path/'arm/raw/002.response.raw').read_text())
 assert json.loads(raw['choices'][0]['message']['tool_calls'][0]['function']['arguments'])==bad
 schema=next(t['function']['parameters']['properties']['work_item'] for t in server.requests[1]['tools'] if t['function']['name']=='submit_candidate')
 assert set(schema['required'])=={'decision','obstacle','operation','claim','evidence_refs','dependencies','completion_checks','remaining_scope','reasoning_need'}
 assert schema['additionalProperties'] is False
 assert set(schema['properties'])==set(schema['required'])
 assert schema['properties']['operation']['enum']==['ground','contrast','close','compose','test']
 assert schema['properties']['evidence_refs']['minItems']==1
 assert schema['properties']['evidence_refs']['items']['anyOf'][0]['maximum']==1


def test_both_arms_receive_same_full_schema_and_valid_object_reaches_abi(tmp_path):
 schemas=[]
 for arm in ('off','on'):
  ledger=tmp_path/(arm+'.json');initial_ledger(ledger)
  with Server([response('execute_python',{'code':'print(1)'},1,arm),response('submit_candidate',{'work_item':WORK,'candidate_source':SOURCE.splitlines()},2,arm)]) as server:
   result=episode(arm,tmp_path/arm,ledger,fixture_url=server.url)
  assert result['frozen'] and result['repair']['failed_submissions']==0
  schema=next(t['function']['parameters']['properties']['work_item'] for t in server.requests[1]['tools'] if t['function']['name']=='submit_candidate')
  schemas.append(schema)
 assert schemas[0]==schemas[1]
 schema=schemas[0]
 jsonschema.Draft202012Validator.check_schema(schema)
 jsonschema.validate(WORK,schema)
 for name,args,expected in CASES:
  with pytest.raises(jsonschema.ValidationError):jsonschema.validate(args.get('work_item'),schema)
 for key in ('decision','obstacle','claim','remaining_scope'):
  assert schema['properties'][key]=={'type':'string','minLength':1,'pattern':r'\S'}
 assert schema['properties']['reasoning_need']['enum']==['needs_computation','needs_observation','does_not_block_decision']
 assert schema['properties']['dependencies']=={'type':'array','minItems':1,'items':{'type':'string','minLength':1,'pattern':r'\S'}}


def test_preserve_inherited_true_step_acceptance_without_coercion(tmp_path):
 ledger=tmp_path/'ledger.json';initial_ledger(ledger)
 work=dict(WORK,evidence_refs=[True],completion_checks=[True])
 with Server([response('execute_python',{'code':'print(1)'},1),response('submit_candidate',{'work_item':work,'candidate_source':SOURCE.splitlines()},2)]) as server:
  result=episode('off',tmp_path/'arm',ledger,fixture_url=server.url)
 assert result['frozen']
 progress=json.loads((tmp_path/'arm/PROGRESS.json').read_text())
 assert progress['work_item']['evidence_refs']==[True] and type(progress['work_item']['evidence_refs'][0]) is bool
 schema=next(t['function']['parameters']['properties']['work_item'] for t in server.requests[1]['tools'] if t['function']['name']=='submit_candidate')
 jsonschema.validate(work,schema)
