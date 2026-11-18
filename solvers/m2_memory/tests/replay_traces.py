"""Bounded replay of selected real inspection calls; never execute game actions."""
import hashlib
import json
from pathlib import Path
import re
import sys
from test_basics import sandbox

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT.parents[1]/'analysis/m2-winners-20261001'
RUN=ARCHIVE/'runs/out-control'
sys.path.insert(0,str(ARCHIVE))
from transcript_stats import blocks

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def call(game,index):
    path=next((RUN/'transcripts').glob(game+'*'))
    count=0; turn=None
    for kind,body in blocks(path.read_text()):
        if kind.startswith('TURN '):turn=int(re.search(r'analysis_step=(\d+)',kind)[1])
        if kind=='TOOL CALL: python':
            count+=1
            if count==index:
                code=re.search(r'<parameter=code>\s*(.*?)</parameter>',body,re.S)[1].strip()
                return code,turn,{'path':str(path.relative_to(ROOT.parents[1])), 'sha256':digest(path),'call':index,'code':code}
    raise AssertionError('missing call')

def state_at(game,turn):
    path=next((RUN/'artifacts').glob(game+'*events*'))
    events=[json.loads(line) for line in path.open()]
    events=[e for e in events if e.get('board') and (e.get('analysis_step') or 0)<turn]
    assert events
    def frame(e):
        return {'grid':e['board'],'ascii':e['board_ascii'],'shape':[len(e['board']),len(e['board'][0])],
                'step':e['action_num'],'level':e['level']}
    return {'current_frame':frame(events[-1]),'history':[{'action':e.get('action_name','INITIAL'),
        'frame':frame(e),'result':{}} for e in events],'valid_actions':[]}, {'path':str(path.relative_to(ROOT.parents[1])),'sha256':digest(path)}

def no_actions(*args,**kwargs):raise AssertionError('inspection replay attempted a game action')
def run(code,state,**kwargs):
    result=sandbox.run_sandboxed_python(code=code,timeout_seconds=3,initial_state=state,action_handler=no_actions,**kwargs)
    assert not result['action_results'], result
    return result

def okay(result):
    assert not result.get('error'),result
    return result['stdout']

reports=[]
# Actual symbol extraction and next-call failure. The only additions are explicit
# JSON encoding at save/load; no inferred globals or implicit stale observation.
c7,turn,source7=call('tr87',7); c8,_,source8=call('tr87',8)
state,events=state_at('tr87',turn)
reference=okay(run(c7+'\n'+c8,state))
first=run(c7,state,kept_functions=[])
functions=[x['source'] for x in first['keepable_functions']]
missing=run(c8,state,kept_functions=functions)
assert "name 'tops' is not defined" in missing['error'],missing
store={}
changed=c7.replace("fr.ascii.split('\\n')",'fr.rows')
assert changed!=c7
save='\nsave_receipt = save_data("tops", [[list(k),v] for k,v in tops.items()])'
load='tops={tuple(k):v for k,v in load_data("tops")["value"]}\n'
a=run(changed+save,state,saved_data=store,kept_functions=[])
b=run(load+c8,state,saved_data=store,kept_functions=[x['source'] for x in a['keepable_functions']])
assert okay(a)+okay(b)==reference
reports.append({'case':'tr87 calls 7 -> 8','sources':[source7,source8,events],
    'baseline_error':missing['error'].splitlines()[-1], 'adaptations':[save,load,"fr.ascii.split('\\n') -> fr.rows"],
    'result':'Two calls match uninterrupted reference stdout exactly', 'saved_bytes':store['tops']['bytes']})
# Preserve the known HistoryEntryView.step failure, but isolate its first line:
# the rest of the recorded snippet contains an unrelated non-advancing while loop.
code,turn,source=call('bp35',61); state,events=state_at('bp35',turn)
selection=code.splitlines()[0]
assert selection=='df=[h.frame for h in history if h.step==52][0]'
result=run(selection+'\nprint(df.ascii)',state)
expected=next(h['frame']['ascii'] for h in state['history'] if h['frame']['step']==52)
assert okay(result).rstrip('\n')==expected.rstrip('\n')
reports.append({'case':'bp35 call 61 first statement only','sources':[source,events],
    'executed_code':selection,'omission':'Rest excluded: unrelated non-advancing loop',
    'result':'Unmodified history selector retrieves archived action-52 board exactly'})
# Real whole inspection snippet, deterministic animation boundary fixture.
code,_,source=call('dc22',175)
timeline={'steps':[{'frame':0,'changed':4}], 'frames':[1,2], 'count':2}
result=run(code,state,animation_handler=lambda req:timeline)
assert okay(result)=="{'frame': 0, 'changed': 4}\nframes 2\ncount 2\n"
reports.append({'case':'dc22 call 175','sources':[source], 'executed_code':code,
    'fixture':timeline,'result':'Unmodified items() inspection works; animation payload is a fixture, not recovered original'})
output={'scope':'Selected CPU inspection replays, not gameplay or causal score evidence', 'cases':reports}
(ROOT/'evidence/trace-replay.json').write_text(json.dumps(output,indent=2)+'\n')
print('PASS: real tr87 two-call data recovery and view equivalence; bp35 history selector; dc22 mapping inspection')
