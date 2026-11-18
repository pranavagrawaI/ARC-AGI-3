"""Assess longer-run host evidence; keep scientific conclusions separate."""
import argparse
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def assess(root):
    errors=[]
    def document(path):
        try:return json.loads(path.read_text())
        except (OSError,ValueError) as error:
            errors.append(f'{path.name}: {error}');return {}
    def rows(path,required=True):
        if not path.exists():
            if required:errors.append(f'missing {path}')
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    result=document(root/'result.json')
    launcher=document(root/'official-launcher-result.json')
    if result.get('status')!='passed' or launcher.get('status')!='passed':errors.append('runtime/launcher did not pass')
    expected=document(HERE/'CHASSIS.json')['contract_sha256']
    sessions=[]
    for session in sorted((root/'solver-workspaces').glob('*')):
        if not session.is_dir():continue
        launch=document(session/'launch.json');stop=document(session/'stop.json');ipc=session/'ipc'
        if launch.get('contract_sha256')!=expected or launch.get('arm')!='control':errors.append('wrong contract or arm')
        if stop.get('reason')!='competition_agent_cleanup':errors.append('unexpected session stop')
        states=rows(ipc/'states.jsonl');transitions=rows(ipc/'transitions.jsonl');calls=rows(ipc/'model_calls.jsonl')
        by_step={r['step']:r for r in transitions}
        for state in states:
            for outcome in state.get('action_results',[]):
                transition=by_step.get(outcome['step'])
                if transition is None:errors.append('result without transition');continue
                if outcome['outcome']=='no_impact' and transition['before']['grid']!=transition['after']['grid']:errors.append('changed grid falsely labelled no_impact')
        delivery=rows(ipc/'state_delivery.jsonl')
        for row in delivery:
            for item in row['delivered']:
                state=json.loads(item)
                if {'turn':state['turn'],'step':state['step']}!=row['latest']:errors.append('stale state delivered')
        if any(c.get('reasoning_tokens',0) for c in calls):errors.append('reasoning returned')
        if (ipc/'ABORT').exists():errors.append('worker ABORT present')
        integrity=rows(ipc/'integrity.jsonl')
        if not integrity:errors.append('missing integrity evidence')
        sessions.append({'session':session.name,'model_calls':len(calls),'actor_calls':sum(not c.get('summarizer',False) for c in calls),'summarizer_calls':sum(bool(c.get('summarizer',False)) for c in calls),'executed_actions':len(transitions),'published_batches':len(states)})
    if not sessions:errors.append('no sessions')
    stops=rows(root/'solver-workspaces/game-stops.jsonl')
    if len(stops)!=1:errors.append('expected exactly one game stop')
    executed=sum(session['executed_actions'] for session in sessions)
    for stop in stops:
        reason=stop.get('reason')
        if reason not in {'win','wall_budget','step_budget'}:errors.append('unknown game stop')
        if stop.get('executed_solver_actions')!=executed:errors.append('game stop action count differs from transitions')
        if stop.get('action_limit')!=100 or executed>100:errors.append('wrong action cap')
        if stop.get('admission_horizon_seconds')!=7200:errors.append('wrong admission horizon')
        if reason=='step_budget' and executed!=100:errors.append('step budget stopped early')
        if reason=='wall_budget' and stop.get('gameplay_elapsed_seconds',0)<7200:errors.append('wall budget stopped early')
    # Reuse the actual packaged runtime's accounting check, including warmup.
    from exact_request_outer_runtime import reconcile_launcher_request_count, server_request_facts
    try:
        reconciliation=reconcile_launcher_request_count(launcher,document(root/'warmup-facts.json'),server_request_facts((root/'server.log').read_text()))
    except (OSError,ValueError,TypeError,RuntimeError) as error:
        errors.append(str(error));reconciliation=None
    return {'status':'INVALID' if errors else 'PASS','errors':errors,'sessions':sessions,'game_stops':stops,'request_reconciliation':reconciliation,'scope':'longer-run host evidence only; no causal solver-improvement claim'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('raw',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=assess(args.raw)
    text=json.dumps(result,indent=2)+'\n'
    if args.output:args.output.write_text(text)
    print(text,end='')
    raise SystemExit(result['status']!='PASS')
