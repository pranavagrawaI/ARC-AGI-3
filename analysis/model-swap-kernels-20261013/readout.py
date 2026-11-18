"""Same time/readout definitions for Tinfield and both saved c14 controls."""
import datetime as dt
import importlib.util
import json
import math
import re
import statistics
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
levels=module('levels_time',REPO/'analysis/pruning-transfer-20261011/readout/levels_time.py')
analyze=module('serving_analyze',REPO/'analysis/model-serving-20261011/analyze.py')
CONTROL={'c14_wave1':REPO/'analysis/model-serving-20261011/wave-1/e2-c14/output',
         'c14_pruning_control':REPO/'analysis/pruning-transfer-20261011/control/output'}
HDR=re.compile(r'^--- analysis_step=(\d+) \| action=(\d+) \| (\d\d:\d\d:\d\d)',re.M)

def thinking_per_action(out,minutes=40):
    reason=generated=responses=actions=missing=0;by_game={}
    for transcript in out.glob('transcripts/*.txt'):
        stem=transcript.stem;heads=HDR.findall(transcript.read_text(errors='replace'))
        if not heads:continue
        secs=[];day=0;previous=None
        first=dt.datetime.strptime(heads[0][2],'%H:%M:%S')
        for step,action,stamp in heads:
            t=dt.datetime.strptime(stamp,'%H:%M:%S')
            if previous and t<previous:day+=1
            secs.append((int(step),int(action)-1,(t-first).total_seconds()+86400*day));previous=t
        clocks={s:t for s,a,t in secs};r=g=n=m=0
        requests_path=out/(stem+'_requests.jsonl')
        if requests_path.exists():
            for line in requests_path.open():
                row=json.loads(line)
                if row.get('event')!='response':continue
                when=clocks.get(row.get('analysis_step'))
                if when is None:m+=1;continue
                if when>minutes*60:continue
                u=row.get('usage',{});v=u.get('reasoning_tokens')
                if v is None:m+=1;continue
                assert math.isfinite(float(v)) and v>=0
                r+=int(v);g+=int(u.get('completion_tokens',0));n+=1
        action_count=0
        ep=out/'artifacts'/(stem+'_events.jsonl')
        if ep.exists():
            for line in ep.open():
                row=json.loads(line)
                if row.get('type')!='action':continue
                later=[t for s,a,t in secs if a>=row['action_num']]
                when=later[0] if later else secs[-1][2]
                if when<=minutes*60:action_count+=1
        reason+=r;generated+=g;responses+=n;actions+=action_count;missing+=m
        by_game[stem[:4]]={'reasoning_tokens':r,'completion_tokens':g,'responses':n,'actions':action_count,'unmapped_or_missing_reasoning_usage':m}
    return {'window_game_minutes':[0,minutes],'thinking_tokens':reason,'completion_tokens':generated,'responses_with_reasoning_usage':responses,
            'executed_actions':actions,'thinking_tokens_per_action':reason/actions if actions else None,
            'unmapped_or_missing_reasoning_usage':missing,'by_game':by_game,
            'timing_limit':'Response/action assigned conservatively to transcript turn boundary, as levels_time.py does; no sub-turn timestamp claim.'}

def arm(out):
    curve=levels.run(out);benchmark=json.loads((out/'benchmark.json').read_text());games=benchmark['game_runs']
    serve=analyze.serve_stats(out,5,30)
    return {'games':len(games),'game_ids':sorted(g['game_id'] for g in games),
            'states':{g['game_id']:g['state'] for g in games},'levels_over_time':{str(t):sum(sum(v<=t for v in ls) for ls in curve.values()) for t in [5,10,15,20,25,30,35,40]},
            'per_game_level_minutes':curve,'serve_5_30':serve,'thinking_0_40':thinking_per_action(out)}

def main():
    controls={name:arm(out) for name,out in CONTROL.items()}
    (ROOT/'control-readout.json').write_text(json.dumps(controls,indent=2)+'\n')
    if '--controls-only' in sys.argv:
        print(json.dumps({k:{'curve':v['levels_over_time'],'serve':v['serve_5_30'],'thinking_per_action':v['thinking_0_40']['thinking_tokens_per_action']} for k,v in controls.items()},indent=2));return
    d=ROOT/'tinfield';out=d/'output';collection=json.loads((d/'collection.json').read_text());result={'candidate':'Tinfield-1','kernel':collection['kernel'],'version':collection['version'],'status':'INVALID_NO_QUEUE','smoke_pass':False,'queued':False,'collection':collection}
    if collection['status']!='COMPLETE':
        result['reason']='Kernel did not complete; inspect exact-version logs.'
    else:
        t=arm(out);finite=json.loads((out/'model-swap-finite-answer.json').read_text());lp=finite['choices'][0]['logprobs']['token_logprobs'];assembly=json.loads((out/'model-swap-assembly.json').read_text())
        logs='\n'.join(p.read_text(errors='replace') for p in out.glob('*.log'))
        info=json.loads((out/'model-swap-server-info.json').read_text())
        gates={'READY':bool(re.search(r'READY after \d+s',logs)),'finite_answers':bool(lp) and any(x is not None for x in lp) and all(x is None or math.isfinite(float(x)) for x in lp),
               'acceptance_ge_2':(t['serve_5_30'].get('accept_len_median') or 0)>=2,'no_crashed_games':t['games']==25 and all(s in {'won','gave_up'} for s in t['states'].values()),
               'same_25_public_games':t['game_ids']==controls['c14_wave1']['game_ids'],'levels_at_30_ge_44':t['levels_over_time']['30']>=44,
               'c14_gate_14': 'priority gate active: 14 concurrent streams' in logs,'target_draft_512':assembly['target_experts']==assembly['draft_experts']==512,
               'nonempty_thinking_usage':t['thinking_0_40']['responses_with_reasoning_usage']>0,'serving_window_sufficient':t['serve_5_30'].get('window_s',0)>=600,
               'runtime_c14_config':info.get('max_running_requests')==14 and info.get('context_length')==102400 and info.get('max_mamba_cache_size')==84,
               'runtime_target_format':info.get('quantization')=='modelopt_fp4',
               'runtime_drafter_unchanged_format':info.get('speculative_draft_model_quantization')=='compressed-tensors'}
        result.update(status='SMOKE_PASS' if all(gates.values()) else 'SMOKE_FAIL',smoke_pass=all(gates.values()),gates=gates,tinfield=t,controls=controls)
    (d/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['controls','tinfield']},indent=2))

if __name__=='__main__':main()
