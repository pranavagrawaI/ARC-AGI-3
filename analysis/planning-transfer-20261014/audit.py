"""Receipt-based paired screen. Empty, fixture and unresolved runs are invalid."""
import argparse
import json
from pathlib import Path

VARIANTS = ('control','planner','probing','both')
VALID = {'solved','token_budget','request_budget','episode_deadline','gave_up'}


def read(path):
    return json.loads(path.read_text())


def native_level_count(actions):
    import importlib.util
    root=Path(__file__).resolve().parents[2]
    path=root/'analysis/planner-qualification-v2-20261013/run_probe.py'
    spec=importlib.util.spec_from_file_location('native_receipt_verifier',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    helpers,_=module.native_context()
    game=helpers['make']()
    if not actions:
        raise ValueError('Empty native action history')
    for action in actions:
        frame=helpers['perform'](game,action)
    return frame.levels_completed


def audit_episode(directory, expected, verify_native=native_level_count):
    required = ['result.json','checkpoint.json','restoration.json','intervention.json','first-request.json','wire.jsonl','actions.json']
    missing = [name for name in required if not (directory/name).exists()]
    if missing:
        return {**expected,'valid':False,'issues':['missing '+name for name in missing]}
    result = read(directory/'result.json')
    checkpoint = read(directory/'checkpoint.json')
    intervention = read(directory/'intervention.json')
    restoration = read(directory/'restoration.json')
    first_request = read(directory/'first-request.json')
    events = [json.loads(line) for line in (directory/'wire.jsonl').read_text().splitlines()]
    issues = []
    if any(result.get(key)!=expected[key] for key in ('checkpoint','seed','variant')):
        issues.append('matrix identity mismatch')
    if result.get('mode')!='live' or result.get('status') not in VALID or result.get('unresolved'):
        issues.append('fixture, failed, or unresolved episode')
    if not checkpoint.get('qualified') or not result.get('qualified_restoration') or not result.get('protected_inputs_unchanged'):
        issues.append('restoration or input identity failed')
    comparisons=checkpoint.get('comparisons',[])
    if not comparisons or not all(row.get('visible_content_identical') and row.get('tools_identical') for row in comparisons):
        issues.append('historical request coverage/divergence')
    if not restoration.get('actions_match'):
        issues.append('restoration action history disagrees')
    if not intervention.get('all_other_messages_identical') or not intervention.get('original_system_prefix_preserved'):
        issues.append('intervention changed undeclared context')
    requests = [row for row in events if row.get('event')=='request']
    responses = [row for row in events if row.get('event')=='response']
    if any(row.get('event')=='failure' for row in events):
        issues.append('failed transport')
    if not requests or len(requests)!=len(responses) or len(requests)!=result.get('new_model_calls'):
        issues.append('request/response coverage mismatch')
    if requests:
        body=requests[0].get('body',{})
        if body.get('messages')!=first_request.get('messages') or body.get('tools')!=first_request.get('tools'):
            issues.append('first wire request differs from declared intervention')
    tokens = 0
    for index, (request,response) in enumerate(zip(requests,responses),1):
        body=request.get('body',{})
        usage=response.get('body',{}).get('usage',{}).get('completion_tokens')
        if request.get('call')!=index or response.get('call')!=index or response.get('status')!=200:
            issues.append('response identity or status mismatch')
        cap=body.get('max_tokens')
        if type(usage) is not int or type(cap) is not int or not 0<=usage<=cap or cap>32768-tokens:
            issues.append('invalid token receipt')
        else:
            tokens+=usage
        if body.get('seed')!=expected['seed']:
            issues.append('sampling seed mismatch')
    if tokens!=result.get('tokens') or tokens>32768 or len(requests)>32:
        issues.append('budget accounting mismatch')
    actions=read(directory/'actions.json')
    prefix=checkpoint.get('prefix_actions',[])
    if not checkpoint.get('actions_match') or actions[:len(prefix)]!=prefix:
        issues.append('action prefix mismatch')
    completed=result.get('levels_completed')
    if type(completed) is not int or completed not in (1,2):
        issues.append('unexpected native level count')
    solved=completed==2
    if solved!=(result.get('status')=='solved'):
        issues.append('terminal/status disagreement')
    try:
        if verify_native(actions)!=completed:
            issues.append('native replay disagrees with reported completion')
    except Exception as error:
        issues.append('native replay failed: '+type(error).__name__)
    return {**expected,'valid':not issues,'issues':issues,'solved':solved,
            'score':result.get('score'),'receipted_tokens':tokens,'new_actions':len(actions)-len(prefix)}


def summarize(rows, matrix):
    keys=[(r['checkpoint'],r['seed'],r['variant']) for r in rows]
    if len(set(keys))!=len(keys):
        raise ValueError('Duplicate episode identity')
    indexed=dict(zip(keys,rows))
    expected={(r['checkpoint'],r['seed'],r['variant']) for r in matrix}
    if set(indexed)!=expected:
        raise ValueError('Matrix coverage differs')
    groups=sorted({(r['checkpoint'],r['seed']) for r in matrix})
    comparisons={}
    for variant in VARIANTS[1:]:
        pairs=[]
        for checkpoint,seed in groups:
            control=indexed[checkpoint,seed,'control']; treatment=indexed[checkpoint,seed,variant]
            valid=control['valid'] and treatment['valid']
            pairs.append({'checkpoint':checkpoint,'seed':seed,'valid':valid,
                          'completion_delta':int(treatment['solved'])-int(control['solved']) if valid else None})
        coverage=sum(pair['valid'] for pair in pairs)
        gains=sum(pair['completion_delta']==1 for pair in pairs)
        losses=sum(pair['completion_delta']==-1 for pair in pairs)
        comparisons[variant]={'pairs':pairs,'coverage':coverage,'gains':gains,'losses':losses,
                             'gate':'INCOMPLETE_OR_INVALID' if not groups or coverage!=len(groups)
                             else 'PASS_SCREEN' if gains>=2 and losses==0 else 'NOT_MET'}
    return {'selected':len(matrix),'valid':sum(row['valid'] for row in rows),
            'comparisons':comparisons,'scope':'Selected-checkpoint adoption/completion screen; not full-game score efficacy'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');args=parser.parse_args()
    root=Path(args.directory);matrix=read(root/'matrix.json')
    rows=[audit_episode(root/row['id'],row) for row in matrix]
    print(json.dumps({'episodes':rows,**summarize(rows,matrix)},indent=2))
