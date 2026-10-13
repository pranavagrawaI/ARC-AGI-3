"""Replay the B1 pin on real requests from the dfranzen control run (no model, no GPU).

For each game, take the latest logged request that still contains the current level's first opener and
has at least 4 user messages after it, then trim it to 60% of its estimated size, with the pin on and off.
Checks per game: the pinned output keeps that opener and the unpinned one does not; the output fits;
roles start system, user; every tool message answers a tool call in the assistant message before it;
user messages stay in chronological order. Prints the pin size distribution.
Run with an interpreter that has the harness dependencies (requests, PIL, numpy).
"""
import glob, json, os, re, statistics, sys, types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT.parents[1] / 'analysis/m2-winners-20261001/runs/out-control'
sys.path.insert(0, str(ROOT / 'src'))
pkg = types.ModuleType('inference.agent'); pkg.__path__ = [str(ROOT / 'src/inference/agent')]
sys.modules['inference.agent'] = pkg
os.environ.update({'ARC3_MEMORY_SECTIONS': 'off', 'LOCAL_ANALYZER_CONTEXT_WINDOW': str((116 + 12) * 1024),
                   'ARC3_CONTEXT_DRAIN_TOKENS': str(58 * 1024), 'MULTIMODAL_CONTEXT': 'current_grid'})
from inference.agent import tool_agent as ta

STEP = re.compile(r'^Current state: step (\d+), level (\d+)\b', re.M)


def text(m):
    c = m.get('content')
    return c if isinstance(c, str) else '\n'.join(str(p.get('text', '')) for p in c or [] if isinstance(p, dict))


def candidate(path):
    best = None
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            ms = r['messages']
            users = [i for i, m in enumerate(ms) if m['role'] == 'user']
            lv = [STEP.search(text(ms[i])) for i in users]
            cur = next((int(x.group(2)) for x in reversed(lv) if x), None)
            first = next((i for i, x in zip(users, lv) if x and int(x.group(2)) == cur), None)
            if first is not None and sum(1 for i in users if i > first) >= 4 and first > 1:
                best = ms
    return best


def valid(ms, a, budget):
    if ms[0]['role'] != 'system' or ms[1]['role'] != 'user':
        return 'roles'
    for i, m in enumerate(ms):
        if m['role'] == 'tool':
            j = i - 1
            while ms[j]['role'] == 'tool':
                j -= 1
            ids = {c.get('id') for c in ms[j].get('tool_calls') or []}
            if ms[j]['role'] != 'assistant' or m.get('tool_call_id') not in ids:
                return f'tool pairing at {i}'
    steps = [int(x.group(1)) for x in (STEP.search(text(m)) for m in ms if m['role'] == 'user') if x]
    if steps != sorted(steps):
        return 'order'
    if a._estimate_request_input_tokens(ms) > budget:
        return 'budget'
    return ''


def main():
    rows, failures = [], []
    for path in sorted(glob.glob(str(LOGS / '*_requests.jsonl'))):
        game = Path(path).name[:4]
        ms = candidate(path)
        if ms is None:
            continue
        a = ta.ToolAgent(model='replay', base_url='http://127.0.0.1:9', api_key='x')
        full = a._estimate_request_input_tokens(ms)
        a._context_budget_tokens = int(full * 0.6)
        os.environ['ARC3_PIN_LEVEL_OPENING_TOKENS'] = '12288'
        span = a._level_opening_pin(ms)
        opener = text(ms[span[0]])[:200] if span else None
        out = {}
        for knob in ('12288', '0'):
            os.environ['ARC3_PIN_LEVEL_OPENING_TOKENS'] = knob
            out[knob] = a._trim_messages_for_context(ms)
        kept_on = any(text(m)[:200] == opener for m in out['12288']) if opener else False
        kept_off = any(text(m)[:200] == opener for m in out['0']) if opener else False
        err = valid(out['12288'], a, a._context_budget_tokens)
        pin_tok = (a._estimate_request_input_tokens([ms[0], *ms[span[0]:span[1]]]) - a._estimate_request_input_tokens([ms[0]])) if span else 0
        rows.append((game, full, pin_tok, span is not None, kept_on, kept_off, err))
        # kept_off can be true when the opening is recent enough to survive the trim anyway;
        # that game just does not discriminate
        if span is not None and (not kept_on or err):
            failures.append((game, kept_on, kept_off, err))
    for r in rows:
        print(f'{r[0]} request {r[1]:>6} tok  pin {r[2]:>6} tok  pinned={r[3]} kept_on={r[4]} kept_off={r[5]} {r[6] or "ok"}')
    pins = [r[2] for r in rows if r[3]]
    print(f'discriminating (dropped when off): {sum(1 for r in rows if r[3] and not r[5])}')
    print(f'games {len(rows)}  pinned {len(pins)}  pin tokens median {statistics.median(pins) if pins else 0:.0f} max {max(pins) if pins else 0}')
    if failures or not pins:
        raise SystemExit(f'REPLAY FAILED: {failures}')
    print('REPLAY PASS')


def sizes():
    """Token cost of the current level's first 1..4 user-message segments, per game."""
    firsts = []
    for path in sorted(glob.glob(str(LOGS / '*_requests.jsonl'))):
        ms = candidate(path)
        if ms is None:
            continue
        a = ta.ToolAgent(model='replay', base_url='http://127.0.0.1:9', api_key='x')
        users = [i for i, m in enumerate(ms) if m['role'] == 'user']
        lv = [STEP.search(text(ms[i])) for i in users]
        cur = next(int(x.group(2)) for x in reversed(lv) if x)
        first = next(i for i, x in zip(users, lv) if x and int(x.group(2)) == cur)
        later = [i for i in users if i > first]
        base = a._estimate_request_input_tokens([ms[0]])
        costs = [a._estimate_request_input_tokens([ms[0], *ms[first:b]]) - base for b in later[:4]]
        roles = [m['role'] for m in ms[first:later[0]]]
        firsts.append(costs[0])
        print(Path(path).name[:4], 'segments (cumulative tok)', costs, ' first segment messages', len(roles), 'tools', roles.count('tool'))
    print('median first segment', statistics.median(firsts))


if __name__ == '__main__':
    sizes() if '--sizes' in sys.argv else main()
