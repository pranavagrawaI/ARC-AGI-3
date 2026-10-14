"""One pre-recorded Y- prediction per frozen model, checked against the engine."""
import copy
import json
from pathlib import Path
from qualify_models import ARCHIVE, EPISODES, ROOT, load_module, model, native_prefix, sha

HERE=Path(__file__).resolve().parent


def main():
    manifest_path=HERE/'probe-manifest.json'
    if manifest_path.exists():
        raise FileExistsError('Do not overwrite probe evidence')
    source=ROOT/'solvers/m2_planning_tools/overlay/inference/utils/prospective_probe.py'
    files=[source,Path(__file__)]+[ARCHIVE/episode/'restoration.json' for episode in EPISODES]
    manifest={'episodes':list(EPISODES),'action':['s','Y'],
              'scope':'Retrospective frozen-model diagnostic; action family selected from the earlier audit, not a model-adoption experiment',
              'inputs':{str(path.relative_to(ROOT)):sha(path) for path in files}}
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    probe=load_module('native_prospective_probe',source).prospective_probe
    results=[]
    for episode in EPISODES:
        namespace=model(episode)
        helpers,game,frame=native_prefix(episode)
        holder=[frame]
        grow=EPISODES[episode]
        path=HERE/'evidence'/(episode+'-probe.json')
        stages=[]
        def observe():
            return helpers['state_from_board'](holder[0].frame[-1])
        def predict(before,action):
            return (namespace['sim_step'](copy.deepcopy(before),'s','Y',namespace['static_set']())
                    if grow=='sim_step' else namespace['shrinkS'](copy.deepcopy(before),'Y'))
        def record(row):
            stages.append(row['status'])
            path.write_text(json.dumps(row,indent=2)+'\n')
        def execute(action):
            acknowledged=json.loads(path.read_text())
            if acknowledged['status']!='PREDICTED' or acknowledged['actual'] is not None:
                raise RuntimeError('Prediction was not committed before native action')
            holder[0]=helpers['perform'](game,{'id':'ACTION6','data':{'x':36,'y':57}})
            return {'executed_count':1,'level_completed':holder[0].levels_completed>1}
        try:
            row=probe(['s','Y'],observe=observe,predict=predict,state_key=namespace['keyS'],execute=execute,record=record)
            results.append({'episode':episode,'status':row['status'],'stages':stages,'matched':row['matched']})
        except Exception as error:
            results.append({'episode':episode,'status':'ERROR','stages':stages,'error':repr(error)})
    if any(sha(ROOT/relative)!=expected for relative,expected in manifest['inputs'].items()):
        raise RuntimeError('Frozen probe input changed')
    report={'scope':manifest['scope'],'selected':len(results),'executed':sum('ACTED' in row['stages'] for row in results),
            'matched':sum(row.get('matched') is True for row in results),
            'mismatched':sum(row.get('matched') is False for row in results),
            'failed':sum(row['status']=='ERROR' for row in results),'rows':results,'model_calls':0}
    (HERE/'PROBE-RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
