"""Audit saved qualification receipts; never makes network calls or changes the queue."""
import ast
from collections import Counter
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]


def history(run):
    value=run['history']
    return value if isinstance(value,list) else ast.literal_eval(value)


def audit(output, collection, require_native=True):
    output=Path(output)
    target=json.loads((output/'pruning/target.json').read_text())
    draft=json.loads((output/'pruning/draft.json').read_text())
    info=json.loads((output/'pruning/server-info-start.json').read_text())
    config=json.loads((output/'pruning/target-config.json').read_text())['text_config']
    games=json.loads((output/'benchmark.json').read_text())['game_runs']
    reference=json.loads((PROJECT/'analysis/pruning-transfer-20261011/candidate2/output/benchmark.json').read_text())['game_runs']
    log=(output/'kernel.log').read_text(errors='replace')
    actions=sum(len(history(g)) for g in games)
    gates={
        'remote_complete':collection['status']=='COMPLETE',
        'target_bank_448':target.get('complete') is True and target.get('target_experts')==448,
        'target_mask_unchanged':target.get('mask_sha256')=='d7d6269cc0c38026edfcdc4d00fecd5a84227ce121a4c4f58d7687f489ad9ef6',
        'tensor_counts':target.get('counts')==dict(kept=193536,dropped=27648,routers=48),
        'drafter_bank_512':draft.get('draft_experts')==512,
        'ten_active_experts':config['num_experts_per_tok']==10,
        'same_public_25':len(games)==25 and {g['game_id'] for g in games}=={g['game_id'] for g in reference},
        'no_crashed_games':all(g['state'] in {'won','gave_up','cancelled'} for g in games),
        'native_actions_present':actions>0,
        'benchmark_executed':'Starting benchmark...' in log,
        'active_stream_marker':'priority gate active: 18 concurrent streams' in log,
        'submission_output_present':(output/'submission.parquet').is_file() and (output/'submission.parquet').stat().st_size>0,
    }
    expected_server={'max_running_requests':18,'context_length':102400,'dtype':'bfloat16',
                     'quantization':'auto-round','kv_cache_dtype':'fp8_e4m3',
                     'mem_fraction_static':0.96,'speculative_num_steps':3,
                     'speculative_num_draft_tokens':4,'speculative_eagle_topk':1,
                     'max_mamba_cache_size':108}
    # Installed wheel naming is checked against the parent receipt, never guessed.
    prior_info=json.loads((PROJECT/'analysis/pruning-transfer-20261011/candidate2/output/pruning/server-info-start.json').read_text())
    for key,value in expected_server.items():
        if key not in prior_info:
            continue
        assert prior_info[key]==value,(key,prior_info[key],value)
        gates['server_'+key]=info.get(key)==value
    totals={}
    for game in games:
        gid=game['game_id'];events_path=output/'artifacts'/f'{gid}_p0_events.jsonl'
        events=[json.loads(line) for line in events_path.read_text().splitlines() if line]
        actual=[e for e in events if e.get('type')=='action']
        gates['action_history_'+gid]=len(actual)==len(history(game))
        gates['event_levels_'+gid]=bool(events) and int(events[-1]['score'])==int(game['levels_completed'])
        gates['fresh_start_'+gid]=bool(events) and events[0].get('action_num',0)==0 and events[0]['score']==0
        totals[gid]=dict(levels=game['levels_completed'],actions=len(history(game)),state=game['state'])
    if require_native:
        native=json.loads((HERE/'native-replay.json').read_text())
        gates['independent_native_replay']=native['passed'] and native['games']==25 and native['actions']==actions
    failed=[k for k,v in gates.items() if not v]
    return dict(status='PASS' if not failed else 'FAIL',gates=gates,failed=failed,games=len(games),
                levels=sum(g['levels_completed'] for g in games),mean_score=sum(g['final_score'] for g in games)/len(games),
                native_actions=actions,states=dict(Counter(g['state'] for g in games)),per_game=totals,
                interpretation='Operational qualification only; no comparative score claim.')


def main():
    response=json.loads((HERE/'launch-response.json').read_text())
    collection=json.loads((HERE/'collection.json').read_text())
    assert collection['kernel_id']==response['kernelId'] and collection['version']==response['versionNumber']
    assert collection['bound_current_identity_before_and_after']
    for name,digest in json.loads((HERE/'launch-freeze.json').read_text()).items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest,('freeze',name)
    for item in collection['downloaded']:
        path=HERE/'output'/item['file']
        assert path.stat().st_size==item['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'],item['file']
    spec=importlib.util.spec_from_file_location('reap_prepared_branch',HERE/'prepare.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    nb=json.loads(next((HERE/'kernel').glob('*.ipynb')).read_text())
    assert module.competition_guard(nb) is True
    assert module.runtime_config(nb,True)['per_game_s']==31920
    assert module.deadline(nb,True) is None
    result=audit(HERE/'output',collection)
    result.update(utc=dt.datetime.now(dt.timezone.utc).isoformat(),kernel_id=response['kernelId'],
                  version=response['versionNumber'],artifact_hashes_verified=len(collection['downloaded']),
                  frozen_source_verified=True,competition_branch_cpu_verified=True)
    (HERE/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['gates','per_game']},indent=2))
    if result['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':
    main()
