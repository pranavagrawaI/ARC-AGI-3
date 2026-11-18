"""Automatic receipt reconciliation; never promotes or launches another run."""
from __future__ import annotations
import argparse
import collections
import json
from pathlib import Path


def jsonl(path):return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def audit(directory):
    directory=Path(directory);root=directory/'output';issues=[];games=[]
    report={'interpretation':'Qualification only; score is not a comparative result.',
            'mechanism_correctness':'Unmeasured; requires prospective prediction/observation assessment.',
            'issues':issues,'games':games}
    def require(condition,message):
        if not condition:issues.append(message)
    receipt=json.loads((directory/'collection-receipt.json').read_text())
    require(receipt.get('critical_complete',False),'critical evidence missing')
    paths=[root/'tycho-dfranzen'/name for name in ['benchmark.json','candidate-config.json']]
    if all(p.exists() for p in paths):
        benchmark,config=[json.loads(p.read_text()) for p in paths]
        report['config']=config
        expected=config['game_count'];require(len(benchmark['game_runs'])==expected,'run count differs from configuration')
        if directory.name.startswith('gpu-'):
            require(receipt.get('provider_samples_complete',False),'provider request/response samples missing')
            require(config['mode']=='small' and config['tycho_mode']=='orchestrator' and config['treatment']=='control','wrong solver mode')
            require(config['per_game_seconds']==2400 and config['per_game_action_cap'] is None,'wrong game horizon')
            require({g.split('-')[0] for g in config['game_names']}=={'ar25','ls20','sb26'},'wrong game set')
            start=json.loads((root/'qualification-start.json').read_text())
            require(len(start['gpu_inventory'])==1 and 'RTX PRO 6000' in start['gpu_inventory'][0],'wrong hardware')
        action_ids={'RESET':'RESET','UP':'ACTION1','DOWN':'ACTION2','LEFT':'ACTION3',
                    'RIGHT':'ACTION4','SPACEBAR':'ACTION5','CLICK':'ACTION6','UNDO':'ACTION7'}
        for run in benchmark['game_runs']:
            candidates=list((root/'tycho-dfranzen/hybrid-games').glob(run['game_id']+'-*'))
            if len(candidates)!=1:
                issues.append(f"{run['game_id']}: missing/ambiguous worker directory");continue
            path=candidates[0]
            try:
                compute=json.loads((path/'compute.json').read_text())
                exit_receipt=json.loads((path/'execution-receipt.json').read_text())
                metadata=json.loads((path/'worker-metadata.json').read_text())
                actions=jsonl(path/'host-actions.jsonl');llm=jsonl(path/'llm-trace.jsonl');tools=jsonl(path/'tool-trace.jsonl')
                require([a['sequence'] for a in actions]==list(range(len(actions))),run['game_id']+': action sequence mismatch')
                require(all(a['after']==b['before'] for a,b in zip(actions,actions[1:])),run['game_id']+': successor mismatch')
                require(len(actions)==len(run['history'])==exit_receipt['engine_actions'],run['game_id']+': engine/receipt count mismatch')
                for a,h in zip(actions,run['history']):
                    require(action_ids[a['decision']['action']]==h['action']['id'],run['game_id']+': action mismatch')
                    if a['decision']['action']=='CLICK':
                        require({'x':a['decision']['col'],'y':a['decision']['row']}==h['action']['data'],run['game_id']+': click mismatch')
                require(sum(x['completion_tokens'] for x in llm)==compute['completion_tokens'],run['game_id']+': trace tokens mismatch')
                require(sum(x['generated_tokens'] for x in run['history'])+run['final_generated_tokens']==compute['completion_tokens'],run['game_id']+': benchmark tokens mismatch')
                require(len(llm)==compute['calls'],run['game_id']+': call count mismatch')
                require(run['state']!='crashed',run['game_id']+': crashed; diagnose before comparison')
                if directory.name.startswith('gpu-'):
                    require(metadata['client_kind']=='live' and metadata['mode']=='orchestrator',run['game_id']+': wrong worker client/mode')
                requests=[t for t in tools if t['name']=='invoke_builder']
                row={'game':run['game_id'],'state':run['state'],'levels':run['levels_completed'],
                     'score':run['final_score'],'actions':len(actions),'completion_tokens':compute['completion_tokens'],
                     'roles':compute['roles'],'builder_requests':len(requests),
                     'builder_deadline_denials':sum('BuilderDeadlineUnavailable' in str(t.get('result')) for t in requests),
                     'worker_returncode':exit_receipt['worker_returncode'],
                     'stop_reason':exit_receipt.get('stop_reason'),'failure':exit_receipt['failure'],
                     'unresolved_provider_usage':exit_receipt['unresolved_provider_usage']}
                games.append(row)
            except (OSError,KeyError,ValueError) as error:
                issues.append(run['game_id']+': audit incomplete '+type(error).__name__)
    else:
        issues.append('benchmark/config missing; execution remains unqualified')
    report['receipt_reconciliation']='passed' if not issues else 'failed or incomplete'
    (directory/'AUTOMATIC-AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['Automatic qualification receipt audit: '+report['receipt_reconciliation']+'.',
           'This is not a matched efficacy verdict. Inspect raw model/workspace traces before the next launch.']
    for g in games:
        lines.append(f"{g['game']}: {g['state']}; levels={g['levels']}; score={g['score']}; actions={g['actions']}; Builder requests={g['builder_requests']}; denials={g['builder_deadline_denials']}; roles={g['roles']}")
    lines.extend('ISSUE: '+issue for issue in issues)
    (directory/'AUTOMATIC-RESULT.md').write_text('\n\n'.join(lines)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    args=parser.parse_args();report=audit(args.directory)
    print(json.dumps({'receipt_reconciliation':report['receipt_reconciliation'],'issues':report['issues']}))


if __name__=='__main__':main()
