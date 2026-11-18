"""Reconcile downloaded q4 summaries against actual model/tool/action receipts."""
from pathlib import Path
import collections
import json

ROOT=Path(__file__).resolve().parent/'gpu-q4'
BASE=ROOT/'audit-selected/tycho-dfranzen/hybrid-games'

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]

def audit():
    benchmark=json.loads((ROOT/'output/tycho-dfranzen/benchmark.json').read_text())
    runs={r['game_id']:r for r in benchmark['game_runs']}
    results=[]
    action_ids={'RESET':'RESET','UP':'ACTION1','DOWN':'ACTION2','LEFT':'ACTION3',
                'RIGHT':'ACTION4','SPACEBAR':'ACTION5','CLICK':'ACTION6','UNDO':'ACTION7'}
    for directory in sorted(BASE.iterdir()):
        game_id=next(game for game in runs if directory.name.startswith(game+'-'))
        run=runs[game_id]
        compute=json.loads((directory/'compute.json').read_text())
        receipt=json.loads((directory/'execution-receipt.json').read_text())
        tools=read_jsonl(directory/'tool-trace.jsonl')
        llm=read_jsonl(directory/'llm-trace.jsonl')
        actions=read_jsonl(directory/'host-actions.jsonl')
        assert [r['sequence'] for r in actions]==list(range(16)),game_id
        assert all(a['after']==b['before'] for a,b in zip(actions,actions[1:])),game_id
        assert len(actions)==len(run['history'])==receipt['engine_actions']==16,game_id
        for a,h in zip(actions,run['history']):
            assert action_ids[a['decision']['action']]==h['action']['id'],game_id
            if a['decision']['action']=='CLICK':
                assert {'x':a['decision']['col'],'y':a['decision']['row']}==h['action']['data'],game_id
        assert actions[-1]['after']['levels_completed']==run['levels_completed'],game_id
        assert sum(x['completion_tokens'] for x in llm)==compute['completion_tokens'],game_id
        assert sum(x.get('generated_tokens',0) for x in run['history'])+run['final_generated_tokens']==compute['completion_tokens'],game_id
        assert len(llm)==compute['calls'],game_id
        errors=[]
        for i,t in enumerate(tools):
            if 'Traceback (most recent call last)' in str(t.get('result')):
                following=tools[i+1] if i+1<len(tools) else None
                errors.append({'call_index':t['call_index'],'source':t['source'],
                    'status_label':t['status'],'error':str(t['result']).strip().splitlines()[-1],
                    'next_tool_call':following['call_index'] if following else None,
                    'next_python_call_has_traceback': 'Traceback (most recent call last)' in str(following.get('result')) if following and following['name']=='run_python' else None})
        requests=[]
        for f in (ROOT/'output/tycho-dfranzen/hybrid-games'/directory.name/'decision-checkpoints/calls').glob('*/request.json'):
            r=json.loads(f.read_text())
            if r['source']=='actor':requests.append(r)
        offered=sum(any(t.get('name',t.get('function',{}).get('name'))=='invoke_builder'
            for t in r['request_body'].get('tools',[])) for r in requests)
        builder_trace=directory/'trigger-builder-trace.jsonl'
        results.append({'game':game_id,'levels_completed':run['levels_completed'],
            'roles':compute['roles'],'output_tokens':compute['completion_tokens'],
            'worker_returncode':receipt['worker_returncode'],'host_failure':receipt['failure'],
            'action_receipts_reconciled':len(actions),'token_receipts_reconciled':True,
            'builder_requests':[{'call_index':t['call_index'],'sequence':t['sequence'],'status':t['status'],'result':t['result']}
                for t in tools if t['name']=='invoke_builder'],
            'builder_deadline_events':[r for r in read_jsonl(builder_trace) if r['event']=='builder_deadline_skipped'] if builder_trace.exists() else [],
            'actor_request_snapshots_checked':len(requests),'snapshots_offering_builder':offered,
            'expected_actor_requests':compute['roles'].get('actor',{}).get('calls',0),
            'python_tracebacks':errors,
            'scribe_tools':dict(collections.Counter(t['name'] for t in tools if t['source']=='scribe')),
            'actor_note_writes':[t['arguments'].get('path') for t in tools if t['source']=='actor' and t['name'] in ('write_file','edit_file')],
            'explicit_donor_helper_calls':[t['call_index'] for t in tools if 'dfr.' in str(t.get('arguments'))]})
    assert len(results)==3
    (ROOT/'AUDIT.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))

if __name__=='__main__':audit()
