"""Identical restarted workers at a preserved game boundary, not Python checkpoints."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
SOURCE=ROOT/'analysis/local-probes/20260911-critic-frequent-ls20-v2/replicate-a'

def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    rows=[json.loads(l) for l in (SOURCE/'actions.jsonl').read_text().splitlines()][:7]
    assert len(rows)==7 and rows[-1]['step']==7
    notes=SOURCE/'critic/snapshots/before-actor-026/knowledge'
    knowledge={p.name:p.read_text() for p in sorted(notes.glob('*.md'))}
    assert 'step size 2 rows vertical / 5 cols horizontal' in knowledge['step_log.md']
    # No later repair, critic responses, source-game code, or evaluator labels.
    data={'kind':'restart from seven replayed actions; no in-process Python state restored',
          'transitions':[{k:r[k] for k in ['step','action','action_data','before','after','levels_completed','state']} for r in rows],
          'retained_actor_notes':knowledge}
    (out/'evidence.json').write_text(json.dumps(data,indent=2)+'\n')
    seed=('Resume after seven previously executed actions. Your fresh worker numbers new actions locally from zero. '
          'The retained notes below are Actor interpretations, not host-certified game semantics. '
          'Full before/after observations and action data are included. Continue solving from the current state. '
          'No previous Python variables or helpers are restored.\n'+json.dumps(data))
    (out/'seed.txt').write_text(seed)
    (out/'MANIFEST.json').write_text(json.dumps({'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()},
        'source_files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [SOURCE/'actions.jsonl',*notes.glob('*.md')] }},indent=2)+'\n')

def environment(path,fixture):
    from arc_agi import Arcade, OperationMode
    from arcengine import FrameData, GameAction
    manifest=json.loads((fixture/'MANIFEST.json').read_text())
    assert all(hashlib.sha256((fixture/n).read_bytes()).hexdigest()==h for n,h in manifest['files'].items())
    data=json.loads((fixture/'evidence.json').read_text())
    env=Arcade(operation_mode=OperationMode.OFFLINE,environments_dir=str(ROOT/'environment_files'),
        recordings_dir=str(path/'recordings')).make('ls20')
    raw=env.reset()
    for row in data['transitions']:
        assert raw.frame[-1].tolist()==row['before'],('before',row['step'])
        raw=env.step(GameAction[row['action']],data=row['action_data'])
        assert raw.frame[-1].tolist()==row['after'],('after',row['step'])
        assert raw.levels_completed==row['levels_completed'] and raw.state.name==row['state']
    def convert(r):return FrameData(game_id=r.game_id,frame=[a.tolist() for a in r.frame],state=r.state,
        levels_completed=r.levels_completed,win_levels=r.win_levels,guid=r.guid,full_reset=r.full_reset,available_actions=r.available_actions)
    return env,convert(raw),convert
