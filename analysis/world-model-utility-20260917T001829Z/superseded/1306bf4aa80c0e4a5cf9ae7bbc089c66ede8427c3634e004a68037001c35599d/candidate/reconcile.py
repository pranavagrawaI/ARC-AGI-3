"""Read-only integrity and official outcome reconciliation before root promotion."""
import json
from common import read,put,hash_file,digest,encoded

def actor_outcome(child):
 r=read(child/'RESULT.json');rows=[json.loads(x) for x in (child/'transitions.jsonl').read_text().splitlines()]
 if len(rows)!=8+r['new_actions'] or [x['step'] for x in rows]!=list(range(1,len(rows)+1)):raise ValueError('official transition action count')
 if any(a['after']!=b['before'] for a,b in zip(rows,rows[1:])):raise ValueError('official transition continuity')
 if rows[-1]['after']!=r['observation']:raise ValueError('final official observation')
 card=read(child/'SCORECARD.json')
 if len(card['environments'])!=1 or card['environments'][0]['id']!='cd82-fb555c5d':raise ValueError('scorecard identity')
 env=card['environments'][0]
 if len(env['runs'])!=1:raise ValueError('unexpected reset/run boundary; no favorable row selection')
 run=env['runs'][0]
 if run['actions']!=len(rows) or run['levels_completed']!=r['observation']['levels_completed'] or run['state']!=r['observation']['state']:raise ValueError('scorecard/frame disagreement')
 for n in range(1,r['new_actions']+1):
  join=read(child/f'ACTION-JOIN-{n:03}.json');d=join['decision'];ordinal=d['call']
  if join['transition']!=rows[8+n-1] or hash_file(child/f'ACTOR-INPUT-{ordinal:03}.json')!=d['input_sha256'] or hash_file(child/f'ACTOR-RESPONSE-{ordinal:03}.json')!=d['response_sha256']:raise ValueError('actor transition join')
  actor_input=read(child/f'ACTOR-INPUT-{ordinal:03}.json')
  packet=json.loads(actor_input['messages'][1]['content'])
  if packet['predecision_query_results']!=join['predecision_queries']:raise ValueError('query exposure input mismatch')
  for q in join['predecision_queries']:
   if hash_file(child/f'QUERY-{q["id"]:03}.json')!=q['binding_sha256']:raise ValueError('query source result binding')
 if r['phase']=='control' and r['query_attempts']!=0:raise ValueError('control query exposure')
 return {'status':'reconciled','actions_including_prefix':len(rows),'new_actions':r['new_actions'],'official_score':run['score'],'official_levels_completed':run['levels_completed'],'official_state':run['state'],'scorecard_sha256':hash_file(child/'SCORECARD.json'),'transitions_sha256':hash_file(child/'transitions.jsonl'),'run_guid':run['guid']}
