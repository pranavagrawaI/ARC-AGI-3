"""Offline gate for B1 through the real ToolAgent and sandbox (no model, no GPU).

Each mechanism is checked on AND off, so a pass shows the change is what moves the behaviour:
  1. pin: request-time trims keep the current level's opening turns (and drop them with the knob at 0);
     history stays chronological, starts with a user message and fits the budget;
  2. pin: commit-time trims (token budget and the assistant-turn cap) keep them too;
  3. pin: the server-overflow fallback drops a block outside the pin;
  4. pin: once older levels are gone, successive trims keep an identical [system, pin] prefix (cache-safe);
  5. pin: an opening larger than the cap is not pinned (falls back to today's behaviour);
  6. static turn rules: in the system prompt and absent from openers when on; the reverse when off;
  7. sandbox: `import time` works; the tool timeout reaches 60 s with LOCAL_ANALYZER_TOOL_TIMEOUT=60.
Run with an interpreter that has the harness dependencies (requests, PIL, numpy).
"""
import importlib, json, os, sys, tempfile, types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
pkg = types.ModuleType('inference.agent'); pkg.__path__ = [str(ROOT / 'src/inference/agent')]
sys.modules['inference.agent'] = pkg
os.environ.setdefault('ARC3_MEMORY_SECTIONS', 'off')  # as in the dfranzen notebook
os.environ['LOCAL_ANALYZER_TOOL_TIMEOUT'] = '60'      # as the B1 overlay sets it
os.environ['ARC3_CONTEXT_DRAIN_TOKENS'] = str(1500)
from inference.agent import tool_agent as ta
from inference.agent.runtime_state import Frame, write_runtime_state

checks = {}


def agent_with(budget):
    a = ta.ToolAgent(model='gate', base_url='http://127.0.0.1:9', api_key='x')
    a._context_budget_tokens = budget
    return a


def opener(step, level, tag, size=2400):
    return {'role': 'user', 'content': f'Current state: step {step}, level {level}.\n{tag} ' + 'u' * size}


def game(levels_before=2, turns_l3=10):
    """system, two finished levels, then level 3: an opening turn tagged OPEN3 and many later turns."""
    m = [{'role': 'system', 'content': 'sys'}]
    step = 1
    for lv in range(1, levels_before + 1):
        for t in range(3):
            m += [opener(step, lv, f'L{lv}T{t}'), {'role': 'assistant', 'content': f'reason L{lv}T{t} ' + 'a' * 2400}]
            step += 1
    m += [opener(step, 3, 'OPEN3'), {'role': 'assistant', 'content': 'OPEN3-REASON first look ' + 'a' * 2400}]
    step += 1
    for t in range(turns_l3):
        m += [opener(step, 3, f'L3T{t}'), {'role': 'assistant', 'content': f'reason L3T{t} ' + 'b' * 2400}]
        step += 1
    m.append(opener(step, 3, 'NOW'))
    return m


def text(ms):
    return ' '.join(str(m.get('content')) for m in ms)


def well_formed(ms, a):
    roles = [m['role'] for m in ms]
    order = [int(x.split()[0].rstrip(',')) for x in (str(m['content']).split('step ', 1)[1] for m in ms[1:] if m['role'] == 'user')]
    return (roles[0] == 'system' and roles[1] == 'user' and order == sorted(order)
            and a._estimate_request_input_tokens(ms) <= a._context_budget_tokens)


# 1. request-time trim
for knob in ('12288', '0'):
    os.environ['ARC3_PIN_LEVEL_OPENING_TOKENS'] = knob
    a = agent_with(9000)
    out = a._trim_messages_for_context(game())
    on = knob != '0'
    checks[f'request_trim_pin_{"on" if on else "off"}'] = (('OPEN3-REASON' in text(out)) == on and 'NOW' in text(out)
                                                           and 'L1T0' not in text(out) and well_formed(out, a))

os.environ['ARC3_PIN_LEVEL_OPENING_TOKENS'] = '12288'
a = agent_with(9000); a._trim_messages_for_context(game())
st = a._b1_stats
checks['counters'] = st['requests'] == 1 and st['pinned_requests'] == 1 and st['trims_with_pin'] == 1 and st['pin_tokens_max'] > 0

# 2. commit-time trim, including the assistant-turn cap
os.environ['ARC3_PIN_LEVEL_OPENING_TOKENS'] = '12288'
a = agent_with(9000)
hist = a._persistent_history_messages(game())
checks['commit_trim_pin_on'] = 'OPEN3-REASON' in text(hist) and well_formed([{'role': 'system', 'content': 'sys'}, *hist], a)
os.environ['ARC3_HISTORY_ASSISTANT_TURNS'] = '8'; os.environ['ARC3_HISTORY_TURN_DRAIN'] = '0'
a = agent_with(10 ** 6)
hist = a._persistent_history_messages(game())
n_assist = sum(1 for m in hist if m['role'] == 'assistant')
# the pin's assistant messages count against the cap
checks['turn_cap_keeps_pin_and_cap'] = 'OPEN3-REASON' in text(hist) and n_assist <= 8 and 'L3T5' not in text(hist)
os.environ['ARC3_PIN_LEVEL_OPENING_TOKENS'] = '0'
hist = agent_with(10 ** 6)._persistent_history_messages(game())
checks['turn_cap_drops_opening_when_off'] = 'OPEN3-REASON' not in text(hist)
os.environ.pop('ARC3_HISTORY_ASSISTANT_TURNS'); os.environ.pop('ARC3_HISTORY_TURN_DRAIN')

# 3. overflow fallback drops the oldest block outside the pin. When only the pin and the newest turn are
#    left it falls back to ordinary dropping, the last resort for a request that still does not fit.
os.environ['ARC3_PIN_LEVEL_OPENING_TOKENS'] = '12288'
a = agent_with(10 ** 6)
m = game(levels_before=0, turns_l3=10)
for _ in range(3):
    m = a._force_reduce_messages(m)
checks['overflow_fallback_keeps_pin'] = ('OPEN3-REASON' in text(m) and 'NOW' in text(m)
                                         and 'L3T3' not in text(m) and 'L3T6' in text(m))

# 4. cache-safe prefix: two successive requests share [system, pin...]
a = agent_with(9000)
g = game()
first = a._trim_messages_for_context(g)
g2 = [*g, {'role': 'assistant', 'content': 'reason NOW ' + 'c' * 2400}, opener(99, 3, 'NEXT')]
second = a._trim_messages_for_context(g2)
k = 1 + a._b1_pinned_span
checks['pinned_prefix_stable'] = a._b1_pinned_span > 0 and [m['content'] for m in first[:k]] == [m['content'] for m in second[:k]]

# 5. an opening larger than the cap is not pinned
os.environ['ARC3_PIN_LEVEL_OPENING_TOKENS'] = '500'
a = agent_with(9000)
out = a._trim_messages_for_context(game())
checks['oversized_opening_not_pinned'] = 'OPEN3-REASON' not in text(out) and a._b1_pinned_span == 0

# 6. static turn rules
tmp = Path(tempfile.mkdtemp(prefix='b1gate_'))
state = tmp / 'g001_p0_runtime_state.json'
frame = Frame(grid=((0, 1), (1, 0)), step=5, level=1)
write_runtime_state(state, current_frame=frame, history=[])
LONG = 'Keep tool output compact: use `current_frame.segmentation`'
for knob in ('1', '0'):
    os.environ['ARC3_STATIC_TURN_RULES'] = knob
    a = agent_with(9000); a._ensure_session(state); a._current_valid_actions = ['UP']
    o = a._build_user_prompt(5, valid_actions=['UP'], current_frame=frame, history_entries=[],
                             previous_step_summary={'executed_count': 1, 'level': 1, 'level_transition': False})
    on = knob == '1'
    checks[f'static_rules_{"on" if on else "off"}'] = ((LONG in a._system_prompt) == on and (LONG in o) == (not on)
                                                       and 'call `action(actions)`' in o
                                                       and ('emit exactly the tool-call format' in o) == (not on))
    if on:
        checks['opener_shorter'] = len(o)
    else:
        checks['opener_shorter'] = checks['opener_shorter'] < len(o)

# 7. sandbox import and timeout
a = agent_with(9000); a._ensure_session(state); a._current_valid_actions = ['UP']
r = json.dumps(a._run_python_tool(state, {'code': "import time\nprint('T', time.time() > 0)"}).__dict__, default=str)
checks['sandbox_time_import'] = "T True" in r
checks['tool_timeout_60'] = a._python_timeout == 60 and 'hard time limit of 60 seconds' in a._system_prompt and 'limit of 30 seconds' not in a._system_prompt

print(json.dumps(checks, indent=1))
if not all(checks.values()):
    raise SystemExit('GATE FAILED: ' + ', '.join(k for k, v in checks.items() if not v))
print('GATE PASS')
