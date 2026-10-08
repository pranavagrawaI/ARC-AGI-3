"""Deterministic local replay of logged runs. python replay.py <benchmark.json> checks every game reproduces its logged levels."""
import json, ast, sys, importlib.util
from pathlib import Path
from arcengine import GameAction, ActionInput, ARCBaseGame
ENV=Path.home()/'projects/arc-agi-3-reference/competition-env/environment_files'
def game(gid):
    f=next((ENV/gid[:4]).glob(f'*/{gid[:4]}.py'))
    spec=importlib.util.spec_from_file_location(gid[:4],f); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return [v for v in vars(m).values() if isinstance(v,type) and issubclass(v,ARCBaseGame) and v is not ARCBaseGame][0]()
P=lambda v: v if isinstance(v,list) else ast.literal_eval(v)
if __name__=='__main__':
  b=json.load(open(sys.argv[1]))
  for r in b['game_runs']:
    gid=r['game_id']; h=P(r['history']); g=game(gid); fr=g.perform_action(ActionInput(id=GameAction.RESET))
    for x in h:
        a=x['action']; d=a.get('data') or {}
        fr=g.perform_action(ActionInput(id=GameAction[a['id']], data=d))
    print(gid[:4], 'logged', r['levels_completed'], 'replayed', fr.levels_completed, fr.state.name, 'OK' if int(r['levels_completed'] or 0)==fr.levels_completed else 'MISMATCH')
