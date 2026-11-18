"""Compact common actor. Only structured tools act; generated Python never runs here."""
import json,time
from common import MODEL,CAPS,put,journal,encoded,digest
from game import compact,parse_action
from protocol import assistant_message
from codec import messages,EXPLANATION
SYSTEM='''Play this unfamiliar visual grid game to complete official levels and win, within100 new actions. Pixel changes and model agreement are not official progress.
Return exactly ONE tool call. act commits one legal action; CLICK syntax is "CLICK x y". Available actions are in the official observation. RESET counts as an action and disables the model. retrieve reads up to TWO whole already-executed transitions, never future evidence or a game action. query_model predicts a hypothetical legal action without executing it. You may choose a different action after a query. Model outputs are untrusted, possibly wrong; errors/abstentions are possible.
Budget:120 reasoning-off requests,1024 completion tokens each,100 new actions. Treatment additionally has the exact previously frozen model, one initial RIGHT query and at most19 optional queries. Control uses observations/retrieval.
Fixed context: current observation, last TWO full transitions, all chronological action/status/level summaries, latest notes, previous assistant/tool exchange, and ALL current-state query results. Older whole transitions are available in retrieval batches of2. No silent truncation. Every tool requires replacement notes (maximum64 Unicode characters); preserve useful information there. Query evidence precedes commitment.
'''
SYSTEM += '\n'+EXPLANATION

def tool(name,description,props,required):return {'type':'function','function':{'name':name,'description':description,'parameters':{'type':'object','properties':props,'required':required,'additionalProperties':False}}}
NOTES={'type':'string','maxLength':64}
ACT=tool('act','Commit one official action.',{'action':{'type':'string'},'notes':NOTES},['action','notes'])
RETRIEVE=tool('retrieve','Read up to2 transitions from your executed prefix only.',{'start_step':{'type':'integer','minimum':1},'count':{'type':'integer','minimum':1,'maximum':2},'notes':NOTES},['start_step','count','notes'])
QUERY=tool('query_model','Query a legal hypothetical action before deciding; no game step.',{'action':{'type':'string'},'notes':NOTES},['action','notes'])

def row_packet(row):return {**{k:v for k,v in row.items() if k not in ('before','after')},'before':compact(row['before']),'after':compact(row['after'])}

def run(game,phase,model,transport,out,expected,action_limit=100,soft=1800):
 if phase not in ('control','treatment'):raise ValueError('phase')
 if game.prefix()!=expected:raise ValueError('matched prefix mismatch')
 from model import Model
 started=time.monotonic();new_actions=0;calls=0;notes='';exchange=[];exposures=[];reason='action_cap';milestones=[];action_milestones=[]
 if phase=='treatment':
  model=Model(model,game.out/'prefix.jsonl',out)
  if 'RIGHT' in game.current['available_actions']:
   forced=model.query('RIGHT',game.current,len(game.rows),forced=True);exposures=[forced]
  else:raise ValueError('mandatory RIGHT unavailable after prefix')
 initial_evidence={'observation':compact(game.current),'prefix':[row_packet(r) for r in game.rows]}
 put(out/'INITIAL-EVIDENCE.json',initial_evidence)
 try:
  while new_actions<action_limit and game.current['state']=='NOT_FINISHED':
   if time.monotonic()-started>=soft-20:reason='soft_time_cap';break
   if calls>=120:reason='request_cap';break
   packet={'objective':'complete official levels; win if possible','current':compact(game.current),'recent_transitions':[row_packet(r) for r in game.rows[-2:]],'executed_summary':[[r['step'],r['action'],r['after']['state'],r['after']['levels_completed']] for r in game.rows], 'notes':notes,'new_actions':new_actions,'requests_used':calls,'model_access':phase=='treatment','query_attempts':model.query_count if model else 0,'query_executions':sum(q['status'] in ('predicted','error') for q in model.queries) if model else 0,'predecision_query_results':exposures}
   tools=[ACT,RETRIEVE]+([QUERY] if phase=='treatment' and model.query_count<20 else [])
   body={'model':MODEL,'messages':[{'role':'system','content':SYSTEM}]+messages(packet,exchange),'provider':{},'reasoning':{'enabled':False},'tools':tools,'tool_choice':'required','max_tokens':1024}
   call_id=calls+1;request_sha=put(out/f'ACTOR-INPUT-{call_id:03}.json',body)
   response=transport(body);calls+=1;put(out/f'ACTOR-RESPONSE-{call_id:03}.json',response)
   message=assistant_message(response['choices'][0]['message']);tc=message.get('tool_calls')
   if not isinstance(tc,list) or len(tc)!=1:raise ValueError('exactly one actor tool call required; no format retry')
   call=tc[0];name=call['function']['name'];args=json.loads(call['function']['arguments'])
   required={'act':{'action','notes'},'query_model':{'action','notes'},'retrieve':{'start_step','count','notes'}}
   if name not in {x['function']['name'] for x in tools} or not isinstance(args,dict) or set(args)!=required[name]:raise ValueError('actor tool schema')
   if not isinstance(args['notes'],str) or len(args['notes'])>64:raise ValueError('actor notes bound')
   notes=args['notes']
   decision={'time_ns':time.time_ns(),'call':call_id,'input_sha256':request_sha,'response_sha256':digest(encoded(response)),'step_before':len(game.rows),'tool':name,'arguments':args,'query_exposure_ids':[q['id'] for q in exposures]}
   put(out/f'DECISION-{call_id:03}.json',decision)
   if name=='retrieve':
    start,count=args['start_step'],args['count']
    if type(start)is not int or type(count)is not int or not 1<=count<=2 or start<1 or start+count-1>len(game.rows):raise ValueError('retrieval future/range')
    result={'transitions':[row_packet(r) for r in game.rows[start-1:start-1+count]]}
   elif name=='query_model':
    parse_action(args['action'],game.current)
    result=model.query(args['action'],game.current,len(game.rows));exposures.append(result)
    # Only current-state query outputs remain in explicit exposure packet; all raw outputs persist.
   else:
    action=args['action'];parse_action(action,game.current)
    if model:model.before_actual(action,game.current,len(game.rows))
    row=game.step(action);new_actions+=1
    join={'decision':decision,'transition':row,'source_sha256':getattr(model,'source_sha',None),'predecision_queries':exposures}
    put(out/f'ACTION-JOIN-{new_actions:03}.json',join)
    if row['after']['levels_completed']>row['before']['levels_completed']:milestones.append({'levels_completed':row['after']['levels_completed'],'new_actions':new_actions,'elapsed_seconds':time.monotonic()-started})
    if new_actions in (20,40,60,80,100):action_milestones.append({'new_actions':new_actions,'levels_completed':row['after']['levels_completed'],'official_state':row['after']['state']})
    if model:model.after_actual(row)
    result={'executed_step':row['step'],'action':action,'observation':compact(row['after'])};exposures=[]
   exchange=[message,{'role':'tool','tool_call_id':call['id'],'content':json.dumps(result,sort_keys=True,separators=(',',':'))}]
  if game.current['state']!='NOT_FINISHED':reason='official_terminal'
 except Exception as e:
  reason='failed_'+type(e).__name__;put(out/'ACTOR-FAILURE.json',{'type':type(e).__name__,'error':str(e)[:2000]})
  raise
 finally:
  put(out/'ACTOR-RESULT.json',{'phase':phase,'stop':reason,'new_actions':new_actions,'requests':calls,'elapsed_seconds':time.monotonic()-started,'observation':game.current,'milestones':milestones,'action_milestones':action_milestones,'query_attempts':model.query_count if model else 0,'query_executions':sum(q['status'] in ('predicted','error') for q in model.queries) if model else 0,'model_active':model.active if model else False,'model_status':model.reason if model else 'control no model','complete_exposure':new_actions==action_limit or game.current['state']!='NOT_FINISHED'})
