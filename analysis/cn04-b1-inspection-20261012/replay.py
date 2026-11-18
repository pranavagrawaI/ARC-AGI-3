"""Verify recorded frames and inspect cn04 connectors with the local engine."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

import numpy as np
from arcengine import ActionInput, ARCBaseGame, GameAction

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SRC = Path('/home/pranav/projects/arc-agi-3-reference/competition-env/environment_files/cn04/2fe56bfb/cn04.py')
records = json.loads((OUT/'records.json').read_text())
spec = importlib.util.spec_from_file_location('cn04_replay', SRC)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
cls = next(v for v in vars(module).values() if isinstance(v, type) and issubclass(v, ARCBaseGame) and v is not ARCBaseGame)


def make():
    game = cls()
    game.perform_action(ActionInput(id=GameAction.RESET))
    game.inspection_win_checks = []
    game.inspection_win_boards = []
    original_check = game.sjwqloivve

    def checked():
        result = original_check()
        if result:
            game.inspection_win_checks.append(sprite_state(game))
            game.inspection_win_boards.append(game.camera.render(game.current_level.get_sprites()).tolist())
        return result

    game.sjwqloivve = checked
    return game


def perform(game, action):
    return game.perform_action(ActionInput(id=GameAction[action['id']], data=action.get('data') or {}))


def sprite_state(game):
    result = []
    for s in game.current_level.get_sprites():
        if not s.is_visible:
            continue
        raw = game.hlxyvcmpk[s.name]
        old = s.pixels
        s.pixels = raw
        rendered = s.render().copy()
        s.pixels = old
        contacts = np.argwhere((rendered==8) | (rendered==13))
        matched = len(game.iahpylgry.get(s.name, set()))
        joined = len(game.ydfurpdwv.get(s.name, set()))
        result.append({'name':s.name, 'x':s.x, 'y':s.y, 'rotation':s.rotation,
                       'contacts':len(contacts), 'matched':matched,
                       'visually_joined':joined,
                       'unmatched':len(contacts)-matched,
                       'contact_positions_xy':[[int(x+s.x),int(y+s.y)] for y,x in contacts],
                       'typed_contacts':[{'x':int(x+s.x),'y':int(y+s.y),'type':int(rendered[y,x])} for y,x in contacts]})
    return result


audit = {'engine_source':str(SRC), 'engine_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(), 'runs':{}, 'counterfactual':{}}
prefixes, segments = {}, {}
for label, rec in records.items():
    p = ROOT/'solvers/m2_b1/experiment/pair-1'/label/'output'
    b = next(r for r in json.loads((p/'benchmark.json').read_text())['game_runs'] if r['game_id']=='cn04-2fe56bfb')
    events = [json.loads(l) for l in (p/'artifacts/cn04-2fe56bfb_p0_events.jsonl').open()]
    action_events = [e for e in events if e['type']=='action']
    k, end = b['actions_per_level'][0], sum(b['actions_per_level'][:2])
    prefixes[label] = [h['action'] for h in b['history'][:k]]
    segments[label] = [h['action'] for h in b['history'][k:end]]
    game = make()
    states = []
    for i, h in enumerate(b['history'][:end]):
        fr = perform(game, h['action'])
        assert fr.frame[-1] == action_events[i]['board'], (label,i,'frame mismatch')
        assert fr.levels_completed == action_events[i]['score'], (label,i,'score mismatch')
        if k-1 <= i < end:
            # Last action advances immediately to L3; do not count L3 contacts as L2.
            states.append({'index':i+1, 'local_action':i-k+1, 'level_completed':fr.levels_completed,
                           'win_pending':bool(game.rqolqpqwo), 'sprites':sprite_state(game),
                           'completed_level_sprites':game.inspection_win_checks[-1] if fr.levels_completed>1 else None})
    audit['runs'][label] = {'verified_actions':end, 'all_frames_match':True,
                           'final_levels_completed':fr.levels_completed,'states':states,
                           'winning_boards':game.inspection_win_boards,
                           'winning_sprite_states':game.inspection_win_checks}
    print(label,'verified frames',end,'levels',fr.levels_completed)

# Same logged successful L2 sequence, applied after the failed run's L1 prefix.
game = make()
for a in prefixes['b1-r1'] + segments['b1-r2']:
    fr = perform(game,a)
assert fr.levels_completed == 2
audit['counterfactual'] = {'successful_segment_from_failed_prefix_solves':True,
                           'successful_segment_actions':len(segments['b1-r2']),
                           'note':'Offline route replay; no claim that a model intervention would choose this route.'}
(OUT/'engine-replay.json').write_text(json.dumps(audit,indent=2)+'\n')
print('successful L2 route solves from failed-run prefix:',fr.levels_completed==2)
