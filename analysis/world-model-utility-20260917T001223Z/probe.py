import os,sys,json,hashlib,socket
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
os.chdir(OUT)
# Prevent SDK's import-time dotenv loader from consulting any credential file.
import dotenv
dotenv.load_dotenv=lambda *a,**k: False
def deny(*a,**k): raise RuntimeError('network forbidden in feasibility')
socket.socket.connect=deny;socket.create_connection=deny
from arc_agi import Arcade,OperationMode
from arcengine import GameAction
asset=ROOT/'analysis/actor-critic-construction-kaggle-verification-r9/public-assets/files/cd82'
def frame(f):
    assert f is not None
    return {'grid':f.frame[-1].tolist() if hasattr(f.frame[-1],'tolist') else f.frame[-1], 'state':f.state.name,'levels_completed':f.levels_completed,'available_actions':f.available_actions}
def sha(v):return hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()
results=[]
for n,seed in enumerate([0,0,17]):
    arc=Arcade(operation_mode=OperationMode.OFFLINE,environments_dir=str(asset),recordings_dir=str(OUT/f'recordings-{n}'))
    assert [g.game_id for g in arc.available_environments]==['cd82-fb555c5d']
    card=arc.open_scorecard(tags=['local-feasibility-no-model'])
    env=arc.make('cd82-fb555c5d',seed=seed,scorecard_id=card,save_recording=True)
    start=frame(env.reset()); seq=[start]
    for a in [GameAction.ACTION4,GameAction.ACTION2]: seq.append(frame(env.step(a)))
    reset=frame(env.reset())
    rows={'seed_argument':seed,'frames':seq,'reset':reset,'reset_matches_initial':reset==start,'sequence_sha256':sha(seq)}
    (OUT/f'probe-{n}.json').write_text(json.dumps(rows,indent=2)+'\n')
    sc=arc.close_scorecard(card)
    (OUT/f'scorecard-{n}.json').write_text(sc.model_dump_json(indent=2))
    results.append({k:v for k,v in rows.items() if k not in ('frames','reset')})
record=ROOT/'analysis/r10-prep-20260916/exact-collection-20260916/raw/model-runtime/solver-workspaces/arm-B/cd82-fb555c5d-b7azqbdf/ipc/transitions.jsonl'
historical=[json.loads(x) for x in record.read_text().splitlines()]
summary={'runtime':'arc-agi 0.9.9 / arcengine 0.9.3','results':results,'all_sequences_equal':len({x['sequence_sha256'] for x in results})==1,'historical_grid_matches':[seq[0]['grid']==historical[0]['before']['grid'],seq[1]['grid']==historical[0]['after']['grid'],seq[2]['grid']==historical[1]['after']['grid']], 'paid_calls':0,'network_connect_forbidden':True,'dotenv_disabled':True}
(OUT/'PROBE-SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))
