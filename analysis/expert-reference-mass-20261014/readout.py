"""Audit collected raw artifacts; compare equal 900-second gameplay horizons."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import statistics

HERE=Path(__file__).resolve().parent
OUT=HERE/'output/reference-mass'
DECODE=re.compile(r'^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] Decode batch, #running-req: (\d+), #full token: (\d+), full token usage: ([\d.]+).*?accept len: ([\d.]+).*?gen throughput \(token/s\): ([\d.]+), #queue-req: (\d+)')


def parsed(x):return json.loads(x) if isinstance(x,str) else x


def arm(out):
    bench=json.loads((out/'benchmark.json').read_text());start=dt.datetime.fromisoformat(bench['start_time'])
    info=json.loads((out/'server-info.json').read_text())
    runtime={k:info.get(k) for k in ['max_running_requests','context_length','max_mamba_cache_size','quantization','dtype','kv_cache_dtype','speculative_num_steps','speculative_num_draft_tokens','speculative_eagle_topk']}
    games={};errors=[];total_events=0;initial_boards={}
    for g in bench['game_runs']:
        hist=parsed(g['history']);apl=parsed(g['actions_per_level']);offset=(dt.datetime.fromisoformat(g['started_at'])-start).total_seconds()
        times=[];idx=0
        for n in apl[:int(g['levels_completed'] or 0)]:
            idx+=n
            if idx<=0 or idx>len(hist):raise ValueError('level action boundary outside history')
            times.append(offset+hist[idx-1]['wallclock_seconds'])
        events=list(out.glob('artifacts/'+g['game_id']+'*_events.jsonl'))
        if len(events)!=1:errors.append('missing/nonunique events '+g['game_id'])
        else:
            best=0;initials=[]
            for line in events[0].open():
                event=json.loads(line)
                if event.get('type')=='initial':initials.append(event)
                if event.get('type')=='action':best=max(best,int(event.get('score') or 0));total_events+=1
            if best!=int(g['levels_completed'] or 0):errors.append('event completion mismatch '+g['game_id'])
            if len(initials)!=1 or any(e.get('score')!=0 or e.get('level')!=1 or e.get('action_num')!=0 for e in initials):
                errors.append('fresh-state receipt mismatch '+g['game_id'])
            if len(initials)==1:initial_boards[g['game_id']]=hashlib.sha256(json.dumps(initials[0]['board']).encode()).hexdigest()
        games[g['game_id']]={'levels_at_900':sum(t<=900 for t in times),'final_levels':g['levels_completed'],
                             'completion_s':times,'final_score':g.get('final_score') or 0,'state':g['state'],
                             'generated_tokens':g.get('final_generated_tokens')}
    text=(out/'serve.log').read_text(errors='replace');rows=[]
    for line in text.splitlines():
        m=DECODE.match(line)
        if m:rows.append((dt.datetime.fromisoformat(m[1]),int(m[2]),float(m[4]),float(m[5]),float(m[6])))
    duration=weighted=per=0;acc=[];running=[];kv=[]
    for row,nxt in zip(rows,rows[1:]):
        t,r,u,a,g=row
        if not 120<=(t-start).total_seconds()<840:continue
        w=min(10,max(0,(nxt[0]-t).total_seconds()));duration+=w;weighted+=g*w;per+=(g/r*w if r else 0)
        acc.append(a);running.append(r);kv.append(u)
    targets=sorted(set(int(n) for n in re.findall(r'REFERENCE_ROUTING_TARGET layer=(\d+) active=5 reference=10',text)))
    return {'games':games,'levels_at_900':sum(g['levels_at_900'] for g in games.values()),
            'runtime':runtime,'initial_board_hashes':initial_boards,
            'final_levels':sum(g['final_levels'] for g in games.values()),
            'mean_final_score':sum(g['final_score'] for g in games.values())/len(games),
            'event_errors':errors,'action_events':total_events,'decode_window_s':duration,
            'aggregate_tok_s':weighted/duration if duration else None,'per_stream_tok_s':per/duration if duration else None,
            'accept_len_median':statistics.median(acc) if acc else None,'running_median':statistics.median(running) if running else None,
            'kv_median':statistics.median(kv) if kv else None,
            'retraction_lines':sum('retract' in l.lower() and 'server_args' not in l for l in text.splitlines()),
            'candidate_target_layers':targets,'draft_unchanged_marker':'REFERENCE_ROUTING_DRAFT layer=0 active=10 custom=False' in text,
            'start_time':bench['start_time'],'end_time':bench['end_time']}


def main():
    collection=json.loads((HERE/'collection.json').read_text());bad=[]
    for item in collection['downloaded']:
        path=HERE/'output'/item['file']
        if path.stat().st_size!=item['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:bad.append(item['file'])
    result={'kernel':collection['kernel'],'version':collection['version'],'collection_status':collection['status'],
            'raw_hashes_verified':len(collection['downloaded'])-len(bad),'hash_mismatches':bad,'promoted':False,'retry':False}
    required=['control/benchmark.json','candidate/benchmark.json','complete.json']
    missing=[p for p in required if not (OUT/p).is_file()]
    if missing:
        result.update(verdict='INVALID',missing=missing)
        if (OUT/'failure.json').is_file():result['failure']=json.loads((OUT/'failure.json').read_text())
    else:
        control=arm(OUT/'control');candidate=arm(OUT/'candidate');complete=json.loads((OUT/'complete.json').read_text())
        tests=(HERE/'output/kernel.log').read_text(errors='replace')
        install={a:json.loads((OUT/a/'routing-install.json').read_text()) for a in ['control','candidate']}
        gates={'hashes':not bad,'same_25_games':set(control['games'])==set(candidate['games']) and len(control['games'])==25,
               'fresh_harness_scheduler_clock':False,  # Frozen v1 reused the process-wide gate; exact class replay proves expired phase2 clock.
               'same_serving_runtime':control['runtime']==candidate['runtime'],
               'same_initial_boards':control['initial_board_hashes']==candidate['initial_board_hashes'] and len(control['initial_board_hashes'])==25,
               'effective_c14':all(control['runtime'].get(k)==v for k,v in {'max_running_requests':14,'context_length':102400,'max_mamba_cache_size':84,'quantization':'auto-round','speculative_num_steps':3}.items()),
               'event_coverage':not(control['event_errors']+candidate['event_errors']) and control['action_events']>0 and candidate['action_events']>0,
               'target48':candidate['candidate_target_layers']==list(range(48)),'draft10':candidate['draft_unchanged_marker'],
               'control_native':install['control']['control_exact_wheel_bytes'],
               'remote_routing_graph_tests':'Ran 7 tests' in tests and '\nOK\n' in tests,
               'contained':complete['elapsed_s']<3450,
               'adequate_decode':min(control['decode_window_s'],candidate['decode_window_s'])>=300,
               'no_crashed_games':all(g['state'].lower() not in ['crashed','error','failed'] for a in [control,candidate] for g in a['games'].values()),
               'no_retractions':control['retraction_lines']==candidate['retraction_lines']==0}
        ratio=candidate['aggregate_tok_s']/control['aggregate_tok_s'] if control['aggregate_tok_s'] and candidate['aggregate_tok_s'] else None
        lv_ratio=candidate['levels_at_900']/control['levels_at_900'] if control['levels_at_900'] else None
        if not all(gates.values()):verdict='INVALID'
        elif lv_ratio is not None and lv_ratio<=0.75:verdict='NEGATIVE_QUALITY_SCREEN'
        elif ratio>1 and candidate['levels_at_900']>control['levels_at_900'] and candidate['mean_final_score']>=0.9*control['mean_final_score']:verdict='PROMISING_SHORT_SCREEN'
        elif ratio<1 and candidate['levels_at_900']<=control['levels_at_900']:verdict='NEGATIVE_EFFICIENCY_SCREEN'
        else:verdict='AMBIGUOUS_SHORT_SCREEN'
        result.update(verdict=verdict,gates=gates,control=control,candidate=candidate,speed_ratio=ratio,level_ratio=lv_ratio,
                      completion_delta=candidate['levels_at_900']-control['levels_at_900'],runtime_complete=complete,
                      limitations=['one replicate','fixed control-first order','short fresh-game exposure','final score is shutdown snapshot; level gate uses exact900s boundary'])
        result['invalidity_reasons']=['process-wide harness gate reused the control deadline in candidate phase',
                                     'successful GPU unittest stderr was not retained; detailed graph-test receipt unavailable']
        result['remote_test_process_completed_successfully']=True
        result['remote_test_detail']='Frozen launcher check=True reached the following nvcc command; detailed unittest results/skip status were not retained.'
    (HERE/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['control','candidate']},indent=2))


if __name__=='__main__':main()
