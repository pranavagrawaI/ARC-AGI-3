"""Independently replay every logged qualification action on the versioned native games."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from arcengine import ARCBaseGame, ActionInput, GameAction

HERE=Path(__file__).resolve().parent
ENV=Path.home()/'projects/arc-agi-3-reference/competition-env/environment_files'


def main():
    games=json.loads((HERE/'output/benchmark.json').read_text())['game_runs']
    rows=[]
    for run in games:
        short,version=run['game_id'].split('-',1)
        path=ENV/short/version/(short+'.py')
        assert path.is_file(),path
        spec=importlib.util.spec_from_file_location('reap_replay_'+short,path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        types=[v for v in vars(module).values() if isinstance(v,type) and issubclass(v,ARCBaseGame) and v is not ARCBaseGame]
        assert len(types)==1,(short,types)
        game=types[0]()
        frame=game.perform_action(ActionInput(id=GameAction.RESET))
        hist=run['history'] if isinstance(run['history'],list) else ast.literal_eval(run['history'])
        for record in hist:
            a=record['action']
            frame=game.perform_action(ActionInput(id=GameAction[a['id']],data=a.get('data') or {}))
        rows.append(dict(game_id=run['game_id'],actions=len(hist),logged=run['levels_completed'],
                         native=frame.levels_completed,passed=frame.levels_completed==run['levels_completed'],
                         source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    result=dict(passed=len(rows)==25 and all(r['passed'] for r in rows),games=len(rows),
                actions=sum(r['actions'] for r in rows),rows=rows)
    (HERE/'native-replay.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':
    main()
