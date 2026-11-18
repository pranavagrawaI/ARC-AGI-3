"""Fail closed on package/runtime evidence; keep behavioral selection manual."""
import argparse
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent

def assess(root,arm,*,reasoning='off',contingent_arm=None,action_limit=30,admission_seconds=1800):
    errors=[]
    def doc(path):
        try:return json.loads(path.read_text())
        except (OSError,ValueError) as e:errors.append(f'{path}: {e}');return {}
    def rows(path,required=False):
        if not path.exists():
            if required:errors.append(f'missing {path}')
            return []
        try:return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        except ValueError:errors.append(f'corrupt {path}');return []
    result=doc(root/'result.json');counts=doc(root/'request-count-facts.json')
    if result.get('status')!='passed':errors.append('runtime not passed')
    if counts.get('whole_model_request_count_verified') is not True:errors.append('request count unverified')
    lifecycle=result.get('lifecycle',{})
    if any(lifecycle.get(k) is not True for k in ('containment_passed','quiet_window_valid','term_only','log_classification_allowed')):errors.append('lifecycle gates failed')
    manifest=doc(HERE/'CHASSIS.json');sessions=[]
    for path in sorted((root/'solver-workspaces').glob('*')):
        if not path.is_dir():continue
        launch=doc(path/'launch.json');ipc=path/'ipc'
        if launch.get('contract_sha256')!=manifest.get('contract_sha256') or launch.get('solver_arm')!=arm:errors.append('wrong worker identity/arm')
        if contingent_arm is not None and launch.get('contingent_arm')!=contingent_arm:
            errors.append(f"contingent arm not recorded as {contingent_arm}: {launch.get('contingent_arm')!r}")
        calls=rows(ipc/'model_calls.jsonl',True);transitions=rows(ipc/'transitions.jsonl',True)
        actions=rows(ipc/'actions.jsonl',True);experiments=rows(ipc/'decision_experiments.jsonl');checks=rows(ipc/'step_checks.jsonl')
        if not rows(ipc/'integrity.jsonl',True):errors.append('no integrity evidence')
        if (ipc/'ABORT').exists():errors.append('worker abort')
        # The card invalidates a run whose reasoning-off STATE cannot be established,
        # so check the configuration the worker recorded. Observed reasoning is
        # reported as a measurement: an empty think block is not reasoning being on,
        # and aborting the run over it is not a stop reason the card states.
        try:declared=json.loads(launch.get('model_config',{}).get('ARC_LLM_EXTRA_BODY') or '{}')
        except ValueError:declared={};errors.append('model_config extra body is not JSON')
        enabled=declared.get('reasoning',{}).get('enabled')
        thinking=declared.get('chat_template_kwargs',{}).get('enable_thinking')
        if (enabled,thinking)!=(reasoning=='on',reasoning=='on'):
            errors.append(f'reasoning-{reasoning} state not established: {declared!r}')
        observed_reasoning=sum(c.get('reasoning_tokens',0) or 0 for c in calls)
        observed_present=sum(1 for c in calls if c.get('reasoning_present'))
        if not calls or not transitions:errors.append('no completed model call/action')
        by_turn={a['turn']:a for a in actions}
        for check in checks:
            batch=by_turn.get(check['turn'],{})
            if check.get('status')!='matched' and check.get('skipped_actions',0)>0:
                if any(t['turn']==check['turn'] and t['step']>check['step'] for t in transitions):errors.append('action executed after interrupted prefix')
        sessions.append(dict(session=path.name,model_calls=len(calls),actions=len(transitions),batches=len(actions),
            experiments=len(experiments),checks=len(checks),reasoning_tokens=observed_reasoning,
            calls_reporting_reasoning=observed_present,interrupted_actions=sum(c.get('skipped_actions',0) for c in checks),
            final_levels=max((t['after'].get('levels_completed',0) for t in transitions),default=0)))
    if not sessions:errors.append('no sessions')
    stops=rows(root/'solver-workspaces/game-stops.jsonl',True)
    if len(stops)!=1:errors.append('expected one game stop')
    total=sum(s['actions'] for s in sessions)
    for stop in stops:
        if stop.get('action_limit')!=action_limit or stop.get('admission_horizon_seconds')!=admission_seconds:errors.append('wrong budgets')
        if stop.get('executed_solver_actions')!=total or total>action_limit:errors.append('wrong action reconciliation')
        if stop.get('reason') not in ('win','wall_budget','step_budget'):errors.append('unexpected stop')
    return dict(status='INVALID' if errors else 'PASS',errors=errors,arm=arm,reasoning=reasoning,
                contingent_arm=contingent_arm,action_limit=action_limit,sessions=sessions,promotion=False,
                scope='Integration only. Review objective hypotheses, chosen tests and progress in raw IPC; this is not a causal gameplay verdict.')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('raw',type=Path);parser.add_argument('--arm',required=True,choices=['loop','mindset']);parser.add_argument('--output',type=Path)
    parser.add_argument('--reasoning',choices=['off','on'],default='off')
    parser.add_argument('--contingent-arm',choices=['off','on'])
    parser.add_argument('--action-limit',type=int,default=30)
    parser.add_argument('--admission-seconds',type=int,default=1800)
    args=parser.parse_args()
    result=assess(args.raw,args.arm,reasoning=args.reasoning,contingent_arm=args.contingent_arm,
                  action_limit=args.action_limit,admission_seconds=args.admission_seconds)
    text=json.dumps(result,indent=2)+'\n'
    if args.output:args.output.write_text(text)
    print(text);raise SystemExit(result['status']!='PASS')
