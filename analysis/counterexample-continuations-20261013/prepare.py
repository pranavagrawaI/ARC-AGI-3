"""Prepare past-only response tapes and original runtime settings; never launch."""
from pathlib import Path
import ast, hashlib, json, sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=ROOT/'analysis/m2-winners-20261001/audit/dfranzen-repo'
sys.path.insert(0,str(ROOT/'analysis/endpoint-judgment-20261013'))
import screen

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()

def main():
    configurations={
      'b1':('solvers/m2_b1/experiment/pair-1/b1-r1/output',73,'solvers/m2_b1/src'),
      'control':('analysis/m2-winners-20261001/runs/out-control',107,str(SOURCE/'ARC3-Inference'))}
    for name,(source,line,code) in configurations.items():
        p=ROOT/source
        log=p/'s5i5-18d95033_p0_requests.jsonl'
        events=[json.loads(s) for s in log.read_text().splitlines()[:line]]
        requests=[x for x in events if x['event']=='request']
        responses=[x for x in events if x['event']=='response']
        tape=[]
        for a,b,r in zip(requests,requests[1:],responses):
            old={digest(m) for m in a['messages'] if m['role']=='assistant'}
            new=[m for m in b['messages'] if m['role']=='assistant' and digest(m) not in old]
            assert len(new)==1
            tape.append({'request':a,'reply':{'message':new[0],'finish_reason':r['finish_reason'],'usage':r['usage']}})
        assert len(tape)==len(requests)-1
        benchmark=json.loads((p/'benchmark.json').read_text())
        run=next(r for r in benchmark['game_runs'] if r['game_id']=='s5i5-18d95033')
        # Last explicit state marker is the actual action count, distinct from turn header.
        import re
        states=re.findall(r'Current state: step (\d+), level (\d+)\.',json.dumps(requests[-1]['messages']))
        stop=int(states[-1][0])
        # Headers display next action = executed + 1; board message step follows the same convention.
        print(name,'request',line,'state markers',states[-3:],'turn action',requests[-1]['action'])
        config={'name':name,'source':source,'source_log_sha256':hashlib.sha256(log.read_bytes()).hexdigest(),
                'source_line':line,'code':str(ROOT/code),'tape':tape,'branch':requests[-1],
                'benchmark_prefix':run['history'][:requests[-1]['action']-1],
                'number_of_levels':run['number_of_levels'],'baselines':run['base_actions_per_level'],
                'env':screen.notebook_env()}
        config['env'].update({'LOCAL_ANALYZER_MODEL_ID':'flashnext','INFERENCE_ANALYZER_MODEL':'flashnext',
                    'ARC3_MAX_ACTIVE_STREAMS':'0','ARC3_HTTP_RETRIES':'0','ARC3_HTTP_RETRY_INITIAL_SECONDS':'0',
                    'ARC3_WARMUP_ACTION_GAMES':'0','ONLY_RESET_LEVELS':'true','TAAF_RUN_AS_SUBMISSION':'0',
                    'TAAF_MINIMAL_DIAGNOSTICS':'1','MPLBACKEND':'Agg'})
        if name=='b1':config['env']['LOCAL_ANALYZER_TOOL_TIMEOUT']='60'
        (HERE/'inputs').mkdir(exist_ok=True)
        (HERE/'inputs'/f'{name}.json').write_text(json.dumps(config)+'\n')
    b1=json.loads((ROOT/'analysis/failure-audit-20261013/test-inputs/prediction-counterexample.json').read_text())
    reminder=json.loads((ROOT/'analysis/failure-audit-20261013/test-inputs/observation-reminder.json').read_text())
    old=json.loads((ROOT/'experiments/probes/historical-counterexample-20261011/repair-packet.json').read_text())
    example=old['historical_counterexample']
    text='Previously observed transition 33: action S-. Before state: '+str(tuple(example['before']))+'. Observed successor: '+str(tuple(example['observed']))+'. Continue using this observation.'
    (HERE/'inputs/feedback.json').write_text(json.dumps({
       'b1':{'control':'','reminder':reminder['messages'][-1]['content'],'counterexample':b1['messages'][-1]['content']},
       'control':{'control':'','reminder':text,'counterexample':old['feedback']}},indent=2)+'\n')

if __name__=='__main__':main()
