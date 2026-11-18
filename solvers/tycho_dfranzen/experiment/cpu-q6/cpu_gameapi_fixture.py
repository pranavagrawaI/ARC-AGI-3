"""Zero-endpoint packaged rehearsal with actual offline GameAPI instances."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import pickle
import shutil
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--environments',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(args.runtime))
    from taaf.benchmark import Benchmark
    from taaf.game_api import ArcadeSpec,GameAPI
    from taaf.deploy_inline import InlineTarget
    from inference.framework.solver import HarnessSolver
    from hybrid.runner import main as run,write_evidence_index
    os.environ.update(TYCHO_MODE='orchestrator',TYCHO_NOOA_TREATMENT='control',
        HYBRID_RUN_MODE='public25',ARC_LLM_MODEL='scripted-cpu-fixture',
        ARC3_MAX_ACTIVE_STREAMS='0',ARC3_PRIORITY_TAIL_FADE='0',
        LITELLM_LOCAL_MODEL_COST_MAP='True')
    spec=ArcadeSpec(environments_dir=str(args.environments))
    aggregate=args.output/'tycho-dfranzen';aggregate.mkdir(parents=True,exist_ok=False)
    (aggregate/'hybrid-games').mkdir()
    (aggregate/'fixture-inputs').mkdir()
    runs=[]
    for name,decision,action_id in (
        ('ar25',{'action':'UP'},'ACTION1'),
        ('ls20',{'action':'UP'},'ACTION1'),
        ('sb26',{'action':'CLICK','row':0,'col':0},'ACTION6')):
        root=args.output/'cpu-iterations'/name;root.mkdir(parents=True)
        os.environ['TYCHO_NOOA_SCRIPTED_TOOL_CALLS']=json.dumps([
            {'tool':'run_python','arguments':{'code':"print('CPU_COMPONENTS',len(dfr.segment([[0,1],[0,1]])['nodes']))"}},
            {'tool':'take_actions','arguments':{'actions':[decision],'rationale':'scripted GameAPI rehearsal'}}])
        bm=Benchmark(games=[GameAPI(env_name=name,arcade_spec=spec)],job_dir=root,
            solver=HarnessSolver(max_runtime_s_per_game=60,max_actions_per_game=1,analyzer_timeout=30))
        target=InlineTarget();target.actual_run_as_submission=False
        (root/'benchmark.pkl').write_bytes(pickle.dumps(bm))
        (root/'target.pkl').write_bytes(pickle.dumps(target))
        sys.argv=['runner',str(root/'benchmark.pkl'),str(root/'target.pkl')]
        run()
        source=root/'tycho-dfranzen'
        saved=json.loads((source/'benchmark.json').read_text())
        assert len(saved['game_runs'])==1
        item=saved['game_runs'][0]
        assert item['state']=='gave_up' and len(item['history'])==1,item
        assert item['history'][0]['action']['id']==action_id,item
        game=next((source/'hybrid-games').iterdir())
        metadata=json.loads((game/'worker-metadata.json').read_text())
        assert metadata['client_kind']=='scripted' and metadata['mode']=='orchestrator',metadata
        tools=(game/'tool-trace.jsonl').read_text()
        assert 'CPU_COMPONENTS 2' in tools,tools
        actions=[json.loads(line) for line in (game/'host-actions.jsonl').read_text().splitlines()]
        assert len(actions)==1 and actions[0]['decision']['action']==decision['action'],actions
        shutil.copytree(game,aggregate/'hybrid-games'/game.name)
        shutil.copy2(source/'benchmark.json',aggregate/'fixture-inputs'/f'{name}-benchmark.json')
        shutil.copy2(source/'candidate-config.json',aggregate/'fixture-inputs'/f'{name}-config.json')
        runs.append(item)
        print('CPU GameAPI passed:',name,action_id,flush=True)
    # Explicit aggregation of three independent one-game rehearsals. Raw
    # benchmark/config inputs survive, and game runs are copied unmodified.
    saved['game_runs']=runs
    saved['label']='CPU GameAPI fixture: three sequential one-game rehearsals'
    (aggregate/'benchmark.json').write_text(json.dumps(saved,indent=2)+'\n')
    config=json.loads((source/'candidate-config.json').read_text())
    config.update(game_count=3,game_names=['ar25','ls20','sb26'],fixture_runs_sequential=3,
                  fixture_transformation='game_runs concatenated without alteration; raw inputs retained')
    (aggregate/'candidate-config.json').write_text(json.dumps(config,indent=2)+'\n')
    write_evidence_index(aggregate)
    (args.output/'qualification-complete.json').write_text(json.dumps({
        'cpu_fixture_passed':True,'actual_gameapi_games':['ar25','ls20','sb26'],
        'engine_actions':3,'scripted':True,'native_segmentation_passed':True})+'\n')


if __name__=='__main__':main()
