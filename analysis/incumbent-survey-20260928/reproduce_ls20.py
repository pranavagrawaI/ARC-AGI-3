"""Replay one saved inspection-only cell in KF's shipped sandbox, offline.

The variant adds only a finite-board guard. Neither variant can execute a game action.
This is a code-termination probe, not a gameplay or model-quality experiment.
"""
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
DUCK = Path('/home/pranav/projects/arc-agi-3-reference/duck-qwen38')
SOURCE = DUCK / 'runs/flash-next/kf/source-dataset/src/ARC3-Inference'
sys.path.insert(0, str(SOURCE))
from inference.agent.python_tool_sandbox import run_sandboxed_python


def frame(ev):
    return {'ascii': ev['board_ascii'], 'grid': ev['board'], 'step': ev['action_num'],
            'level': ev['level'], 'shape': [64, 64]}


def no_action(actions):
    raise AssertionError('Inspection-only reproduction; game actions forbidden')


def main():
    events_path = DUCK / 'runs/flash-next/out-kf2/artifacts/ls20-9607627b_p0_events.jsonl'
    evs = [json.loads(line) for line in events_path.open()]
    error = next(e for e in json.loads((HERE/'kf-tool-errors.json').read_text())
                 if e['game'] == 'ls20' and 'Tool timed out after' in e['error'])
    before = evs[:error['event_line']]
    snapshots = [e for e in before if e['type'] in ('initial', 'action')]
    state = {'current_frame': frame(snapshots[-1]),
             'history': [{'action': e.get('action_display', 'INITIAL'), 'frame': frame(e)} for e in snapshots],
             'valid_actions': ['UP','DOWN','LEFT','RIGHT'], 'last_action_result': {}}
    original = error['codes'][-1]
    needle = '    def walk(r,c):\n'
    assert original.count(needle) == 1
    bounded = original.replace(needle, needle + '        if not (0 <= r < 64 and 0 <= c < 64): return False\n')
    (HERE/'ls20-timeout-original.py').write_text(original+'\n')
    (HERE/'ls20-timeout-bounded.py').write_text(bounded+'\n')
    result = {'source_event_line': error['event_line'], 'state_action': snapshots[-1]['action_num'],
              'source_sha256': hashlib.sha256(events_path.read_bytes()).hexdigest(),
              'boundary': 'Archived public observations only; no model, GPU, environment, or network calls.',
              'limitations': 'State reconstructed from events; host speed differs from Kaggle; no claim of level improvement.',
              'arms': {}}
    for label, code in [('original', original), ('finite_board_guard', bounded)]:
        start = time.monotonic()
        outcome = run_sandboxed_python(code=code, timeout_seconds=30, initial_state=state, action_handler=no_action)
        result['arms'][label] = {'wall_seconds': time.monotonic()-start, 'result': outcome}
        print(label, json.dumps(result['arms'][label]), flush=True)
    (HERE/'ls20-reproduction.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
