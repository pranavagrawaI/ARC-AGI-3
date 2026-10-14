"""R2 validity checks and equal900s measurement from source-bound raw receipts."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
OUT=HERE/'output/reference-mass'
spec=importlib.util.spec_from_file_location('v1_measurement_helpers',HERE.parent/'readout.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)


def load(path):
    return json.loads(path.read_text()) if path.is_file() else None


def clock_valid(receipt):
    if not receipt or receipt.get('gate_is_fresh') is not True or receipt.get('play_s')!=900:return False
    try:
        start,end,child=[float(receipt[k]) for k in ['clock_start','clock_deadline','child_started']]
        return all(math.isfinite(v) for v in [start,end,child]) and child<=start<end and end-start<=901
    except (KeyError,ValueError,TypeError):return False


def qualification_valid(out):
    receipt=load(out/'routing-tests.json')
    stdout=(out/'routing-tests.stdout.txt').read_text() if (out/'routing-tests.stdout.txt').is_file() else ''
    stderr=(out/'routing-tests.stderr.txt').read_text() if (out/'routing-tests.stderr.txt').is_file() else ''
    return bool(receipt and receipt.get('returncode')==0 and receipt.get('gpu_graph_executed') is True
                and 'REFERENCE_GRAPH_TEST_PASSED' in stdout and re.search(r'Ran 7 tests in ',stderr)
                and re.search(r'^OK$',stderr,re.M) and 'skipped' not in stderr.lower())


def main():
    collection=load(HERE/'collection.json');identity=load(HERE/'launch-identity.json');status=load(HERE/'status.json')
    if not collection:raise RuntimeError('terminal collection receipt is required')
    bad=[]
    for item in collection['downloaded']:
        path=HERE/'output'/item['file']
        if not path.is_file() or path.stat().st_size!=item['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:bad.append(item['file'])
    result={'kernel':collection['kernel'],'version':collection['version'],'collection_status':collection['status'],
            'raw_hashes_verified':len(collection['downloaded'])-len(bad),'hash_mismatches':bad,
            'promoted':False,'automatic_retry':False}
    required=['control/benchmark.json','candidate/benchmark.json','complete.json']
    missing=[name for name in required if not (OUT/name).is_file()]
    if missing:
        result.update(verdict='INVALID',missing=missing,failure=load(OUT/'failure.json'))
    else:
        control,candidate=[helpers.arm(OUT/a) for a in ['control','candidate']]
        complete=load(OUT/'complete.json')
        isolation={a:load(OUT/a/'harness-isolation.json') for a in ['control','candidate']}
        installs={a:load(OUT/a/'routing-install.json') for a in ['control','candidate']}
        clocks_ok=all(clock_valid(r) for r in isolation.values())
        independent=clocks_ok and isolation['control']['pid']!=isolation['candidate']['pid']
        if clocks_ok:independent=independent and isolation['candidate']['clock_start']>isolation['control']['clock_deadline']
        c14={'max_running_requests':14,'context_length':102400,'max_mamba_cache_size':84,
             'quantization':'auto-round','dtype':'bfloat16','kv_cache_dtype':'fp8_e4m3',
             'speculative_num_steps':3,'speculative_num_draft_tokens':4,'speculative_eagle_topk':1}
        phases=complete.get('phases',[])
        native='441a5e992c7299a0a6be1a3268706759062f6a15bf9a0c8350e24def051c4c2f'
        patched='058f22dce20166c43343431c1f42812e8fdad00838a542ca508b0893b5f2472c'
        wheel='d0620216732b82b6efd4f8a78cc9b5f55343a4d3837fdfa3bcbb841b7718a225'
        gates={'hashes':not bad,'collection_version_binding':collection.get('bound_current_identity_before_and_after') is True,
            'remote_source_bound':bool(status and status.get('source_verified') is True and collection['status']=='COMPLETE'),
            'sealed_card':hashlib.sha256((HERE/'EXPERIMENT-CARD.md').read_bytes()).hexdigest()==identity['card_sha256'],
            'same_25_games':set(control['games'])==set(candidate['games']) and len(control['games'])==25,
            'fresh_independent_harness_clocks':independent,
            'same_serving_runtime':control['runtime']==candidate['runtime'],
            'same_initial_boards':control['initial_board_hashes']==candidate['initial_board_hashes'] and len(control['initial_board_hashes'])==25,
            'effective_c14':all(control['runtime'].get(k)==v for k,v in c14.items()),
            'event_coverage':not(control['event_errors']+candidate['event_errors']) and min(control['action_events'],candidate['action_events'])>0,
            'target48':candidate['candidate_target_layers']==list(range(48)),'draft10':candidate['draft_unchanged_marker'],
            'control_native':bool(installs['control'] and installs['control'].get('control_exact_wheel_bytes') is True and installs['control'].get('installed_source_sha256')==native),
            'package_identity':all(i and i.get('wheel_sha256')==wheel and i.get('original_source_sha256')==native and i.get('helper_sha256')==identity['asset_sha256']['routing.py'] for i in installs.values()) and installs['candidate'].get('installed_source_sha256')==patched,
            'remote_routing_graph_tests':qualification_valid(OUT/'control'),
            'contained':complete['elapsed_s']<3450 and complete.get('hard_cap_s')==3600,
            'full_play_exposure':len(phases)==2 and {p['arm'] for p in phases}=={'control','candidate'} and all(900<=p['play_elapsed_s']<=1080 and p.get('benchmark_exists') for p in phases),
            'adequate_decode':min(control['decode_window_s'],candidate['decode_window_s'])>=300,
            'no_crashed_games':all(g['state'].lower() not in ['crashed','error','failed'] for a in [control,candidate] for g in a['games'].values()),
            'no_retractions':control['retraction_lines']==candidate['retraction_lines']==0}
        speed=candidate['aggregate_tok_s']/control['aggregate_tok_s'] if control['aggregate_tok_s'] and candidate['aggregate_tok_s'] else None
        levels=candidate['levels_at_900']/control['levels_at_900'] if control['levels_at_900'] else None
        if not all(gates.values()):verdict='INVALID'
        elif levels is not None and levels<=0.75:verdict='NEGATIVE_QUALITY_SCREEN'
        elif speed>1 and candidate['levels_at_900']>control['levels_at_900'] and candidate['mean_final_score']>=0.9*control['mean_final_score']:verdict='PROMISING_SHORT_SCREEN'
        elif speed<1 and candidate['levels_at_900']<=control['levels_at_900']:verdict='NEGATIVE_EFFICIENCY_SCREEN'
        else:verdict='AMBIGUOUS_SHORT_SCREEN'
        result.update(verdict=verdict,gates=gates,invalidity_reasons=[k for k,v in gates.items() if not v],
            control=control,candidate=candidate,speed_ratio=speed,level_ratio=levels,
            completion_delta=candidate['levels_at_900']-control['levels_at_900'],runtime_complete=complete,
            harness_isolation=isolation,limitations=['one stochastic replicate','fixed control-first order',
            'short fresh-game exposure','final score is shutdown snapshot; levels use exact900s horizon'])
    (HERE/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['control','candidate','harness_isolation']},indent=2))


if __name__=='__main__':main()
