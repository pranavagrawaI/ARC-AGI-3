"""Real ToolAgent regression gate: API documentation and eviction accounting.

No provider is contacted. Message fixtures exercise the same request, commit,
and overflow entry points as live play; the sandbox verifies documented fields.
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import types

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get('M2_BUGFIX_SOURCE', ROOT / 'src'))
sys.path.insert(0, str(SOURCE))
pkg = types.ModuleType('inference.agent')
pkg.__path__ = [str(SOURCE / 'inference/agent')]
sys.modules['inference.agent'] = pkg
os.environ.update({'ARC3_MEMORY_SECTIONS': 'off',
                   'LOCAL_ANALYZER_TOOL_TIMEOUT': '60',
                   'ARC3_HISTORY_ASSISTANT_TURNS': '8',
                   'ARC3_HISTORY_TURN_DRAIN': '0'})
from inference.agent import tool_agent as ta
from inference.agent.runtime_state import Frame, write_runtime_state


def agent(budget):
    result = ta.ToolAgent(model='bugfix-gate', base_url='http://127.0.0.1:9', api_key='unused')
    result._context_budget_tokens = budget
    return result


def messages(turns=16):
    result = [{'role': 'system', 'content': 'sys'}]
    for i in range(turns):
        result.extend([{'role': 'user', 'content': f'Current state: step {i+1}, level 2.\nTURN{i}'},
                       {'role': 'assistant', 'content': f'Observed TURN{i} ' + 'x' * 800}])
    return [*result, {'role': 'user', 'content': f'Current state: step {turns+1}, level 2.\nNOW'}]


checks = {}
for pin in ('12288', '0'):
    os.environ['ARC3_PIN_LEVEL_OPENING_TOKENS'] = pin
    m = messages()
    a = agent(10**6)
    a._b1_path = Path(tempfile.mkdtemp()) / 'counters.json'
    requested = a._trim_messages_for_context(m)
    checks[f'request_no_eviction_{pin}'] = requested == m and a._b1_stats['trims'] == 0
    history = a._persistent_history_messages(m)
    st = a._b1_stats
    checks[f'commit_counted_{pin}'] = (
        len(history) < len(m) - 1 and st['trims'] == 1
        and st.get('commit_trims') == 1 and st['requests'] == 1
        and st.get('commit_messages_dropped') == len(m) - 1 - len(history))
    checks[f'counter_receipt_{pin}'] = json.loads(a._b1_path.read_text()) == st
    before = st['trims']
    a._persistent_history_messages([m[0], *history])
    checks[f'unchanged_commit_not_trim_{pin}'] = st['trims'] == before

    a = agent(2500)
    a._persistent_history_messages(m)
    st = getattr(a, '_b1_stats', {})
    checks[f'commit_token_no_double_count_{pin}'] = (
        st.get('commit_trims') == 1 and st.get('trims') == 1
        and st.get('requests') == 0)

    a = agent(10**6)
    out = a._force_reduce_messages(m)
    st = getattr(a, '_b1_stats', {})
    checks[f'overflow_counted_{pin}'] = (
        len(out) < len(m) and st.get('overflow_trims') == 1
        and st.get('requests') == 0 and st.get('trims') == 1
        and st.get('overflow_messages_dropped') == len(m) - len(out))

# A commit with no actual removal must not count the missing system message
# as an eviction; system is intentionally excluded from persistent history.
a = agent(10**6)
m = messages(1)
a._persistent_history_messages(m)
checks['system_exclusion_not_eviction'] = getattr(a, '_b1_stats', {}).get('trims') == 0
checks['phase_totals_reconcile'] = a._b1_stats['trims'] == sum(
    a._b1_stats[phase + '_trims'] for phase in ('request', 'commit', 'overflow'))
checks['counter_schema_version'] = a._b1_stats['schema_version'] == 2

# A failed commit must release the nested-request suppression flag.
def fail_commit(*args, **kwargs):
    raise RuntimeError('fixture commit failure')
a._persistent_history_messages_pinned = fail_commit
try:
    a._persistent_history_messages(m)
except RuntimeError:
    checks['commit_exception_releases_counter_flag'] = not a._b1_committing_history
else:
    checks['commit_exception_releases_counter_flag'] = False
a._trim_messages_for_context(m)
checks['request_count_recovers_after_commit_exception'] = a._b1_stats['requests'] == 1

# The packed notebook must describe the API the actual subprocess exposes.
with tempfile.TemporaryDirectory() as directory:
    state = Path(directory) / 'g001_p0_runtime_state.json'
    frame = Frame(grid=((0, 1), (1, 0)), step=5, level=1)
    write_runtime_state(state, current_frame=frame, history=[])
    a = agent(10**6)
    a._ensure_session(state)
    a._current_valid_actions = ['UP']
    p = a._system_prompt
    checks['prompt_no_contradictory_fields'] = (
        'exposes only' not in p and 'expose only' not in p
        and 'raw numeric grid is intentionally not exposed' not in p
        and 'raw numeric grid is not available' not in p)
    checks['prompt_exposes_cached_fields'] = all(f'`.{name}`' in p for name in ('grid', 'rows', 'counts'))
    result = a._run_python_tool(state, {'code':
        "assert current_frame.grid == ((0, 1), (1, 0))\n"
        "assert len(current_frame.rows) == 2\n"
        "assert current_frame.counts == {0: 2, 1: 2}\n"
        "assert current_frame.cells(1) == ((0, 1), (1, 0))\n"
        "print('DOCUMENTED_API_OK')"})
    checks['documented_api_real_sandbox'] = 'DOCUMENTED_API_OK' in json.dumps(result.__dict__, default=str)

print(json.dumps(checks, indent=2))
assert all(checks.values()), ', '.join(k for k, v in checks.items() if not v)
print('Bugfix gate PASS')
