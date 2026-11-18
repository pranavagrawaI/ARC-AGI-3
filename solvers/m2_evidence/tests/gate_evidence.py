"""Offline gate for the evidence record through the real ToolAgent and sandbox (no model, no GPU).

Checks, on one game session:
  1. record_evidence/evidence work from model-written Python and the host mirrors the record to artifacts;
  2. after the context trimmer evicts the turn that printed the record, the record is still readable from Python
     and the next opener re-shows it;
  3. a contradicting result is recorded as a revision: the opener and evidence() show the revision, not the
     superseded entry, and evidence(include_revised=True) still has both;
  4. a level start re-shows the previous level's entries; a turn with no trim and no level start does not.
Run with an interpreter that has the harness dependencies (requests, PIL, numpy).
"""
import json, os, sys, tempfile, types
from pathlib import Path

ROOT = Path(os.environ.get('M2_EVIDENCE_ROOT', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / 'src'))
pkg = types.ModuleType('inference.agent'); pkg.__path__ = [str(ROOT / 'src/inference/agent')]
sys.modules['inference.agent'] = pkg
os.environ.setdefault('ARC3_MEMORY_SECTIONS', 'off')  # as in the dfranzen notebook
from inference.agent import tool_agent as ta
from inference.agent.runtime_state import Frame, write_runtime_state

def frame(level, step):
    return Frame(grid=((0, 1), (1, 0)), step=step, level=level)

tmp = Path(tempfile.mkdtemp(prefix='evgate_'))
state = tmp / 'g001_p0_runtime_state.json'
write_runtime_state(state, current_frame=frame(1, 3), history=[])
agent = ta.ToolAgent(model='gate', base_url='http://127.0.0.1:9', api_key='x')
agent._ensure_session(state)
agent._evidence_path = state.parent / f'{state.stem}_evidence.json'
agent._current_valid_actions = ['UP']

def run(code):
    out = agent._run_python_tool(state, {'code': code})
    text = json.dumps(out.__dict__ if hasattr(out, '__dict__') else out, default=str)
    return text

def opener(level, transition=False):
    return agent._build_user_prompt(5, valid_actions=['UP'], current_frame=frame(level, 5),
                                    history_entries=[], previous_step_summary={'executed_count': 1,
                                    'level': level, 'level_transition': transition})

checks = {}
r = run("i=record_evidence(claim='goal: one peg per panel', prediction='level completes when each panel has 1 peg', "
        "test='reduced both panels to 1 peg', observed='no level completion; pegs turned gray', "
        "interpretation='one-peg-per-panel alone is not sufficient; whether pegs must end on the wire cell is open')\nprint('ID', i)")
checks['record_from_python'] = "ID {'id': 1" in r
checks['mirrored_to_artifacts'] = agent._evidence_path.exists() and json.loads(agent._evidence_path.read_text())['entries'][0]['id'] == 1

checks['no_reshow_without_trim'] = 'Your evidence record' not in opener(1)

# Evict: a long history whose oldest turn printed the record, with a budget that forces a drain.
msgs = [{'role': 'system', 'content': 'sys'}]
msgs += [{'role': 'user', 'content': 'turn opener ' + 'x' * 4000}, {'role': 'assistant', 'content': 'E1 recorded: one peg per panel ' + 'y' * 4000}]
for k in range(6):
    msgs += [{'role': 'user', 'content': f'later turn {k} ' + 'z' * 4000}, {'role': 'assistant', 'content': 'ok ' + 'w' * 4000}]
agent._context_budget_tokens = 4000
trimmed = agent._trim_messages_for_context(msgs)
checks['trimmer_evicted_record_turn'] = not any('E1 recorded' in str(m.get('content')) for m in trimmed) and len(trimmed) < len(msgs)
r = run("print('READ', [e['claim'] for e in evidence()])")
checks['readable_after_trim'] = 'one peg per panel' in r
o = opener(1)
checks['reshown_after_trim'] = 'Your evidence record' in o and 'E1 (level 1)' in o
checks['reshow_once'] = 'Your evidence record' not in opener(1)

r = run("record_evidence(claim='goal: pegs must end on the wire cell', prediction='finishing panel A on the wire cell completes the level', "
        "test='replayed the solution ending on (1,6)', observed='level completed', interpretation='confirmed for this layout', revises=1)")
agent._note_history_evicted()
o = opener(1)
checks['revision_shown_superseded_hidden'] = 'E2 (level 1, revises E1)' in o and '- E1 ' not in o
r = run("print('CUR', [e['id'] for e in evidence()], 'ALL', [e['id'] for e in evidence(include_revised=True)])")
checks['current_vs_all'] = "CUR [2] ALL [1, 2]" in r
r = run("record_evidence(claim='x',prediction='x',test='x',observed='x',interpretation='x', revises=1)")
checks['double_revision_rejected'] = 'already revised' in r
r = run("record_evidence(claim='c',prediction='p',test='t',observed='o',interpretation='')")
checks['empty_field_rejected'] = 'interpretation must be a nonempty string' in r
r = run("record_evidence('c','p','t','o','level')")
checks['positional_rejected_with_guidance'] = 'keyword arguments only' in r

o = opener(2, transition=True)
checks['level_start_shows_previous_level'] = 'E2 (level 1' in o
print(json.dumps(checks, indent=1))
if not all(checks.values()):
    raise SystemExit('GATE FAILED: ' + ', '.join(k for k, v in checks.items() if not v))
print('GATE PASS')
