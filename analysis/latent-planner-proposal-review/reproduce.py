"""Exercise the supplied BFS loop, not an implemented planner.

The omitted _step and _out are simple adapters here. No inference, game run,
or production edits. These cases do not depend on callback timeout machinery.
"""
import json
import time
from collections import deque


def key(z):
    return json.dumps(z, sort_keys=True, separators=(',', ':'))


def _step(predict, z, a, budget):
    return predict(z, a)


def _out(status, plan, zs, expanded, seen, errors):
    return dict(status=status, plan=plan, zs=zs, expanded=expanded, seen=seen, errors=errors)


def plan(predict, z0, goal, actions, *, max_nodes=5000, max_depth=12,
         max_seconds=5.0, budget=None):
    deadline, expanded, errors = time.monotonic() + max_seconds, 0, []
    try:
        if goal(z0):
            return _out('already_satisfied', [], [], expanded, 0, errors)
    except Exception as e:
        return _out('goal_error', None, None, 0, 0, [f'goal(z0) raised {type(e).__name__}: {e}'])
    q, seen = deque([(z0, [], [])]), {key(z0)}
    while q:
        if time.monotonic() > deadline:
            return _out('timeout', None, None, expanded, len(seen), errors)
        if expanded >= max_nodes:
            return _out('no_plan_within_budget', None, None, expanded, len(seen), errors)
        z, path, zs = q.popleft()
        expanded += 1
        if len(path) >= max_depth:
            continue
        for a in actions:
            try:
                nz = _step(predict, z, a, budget)
            except Exception as e:
                if len(errors) < 5:
                    errors.append(f'predict({a}) raised {type(e).__name__}: {e}')
                continue
            k = key(nz)
            if k in seen:
                continue
            seen.add(k)
            try:
                hit = goal(nz)
            except Exception as e:
                return _out('goal_error', None, None, expanded, len(seen),
                            errors + [f'goal raised {type(e).__name__}: {e}'])
            if hit:
                return _out('found', path + [a], zs + [nz], expanded, len(seen), errors)
            q.append((nz, path + [a], zs + [nz]))
    return _out('no_plan_exhausted', None, None, expanded, len(seen), errors)


def broken(z, a):
    raise ValueError('missing dynamics')


def mutating(z, a):
    z['x'] += a
    return z


results = {}
results['depth_cutoff'] = plan(lambda z,a: {'x':z['x']+1}, {'x':0}, lambda z:z['x']==2, ['UP'], max_depth=1)
assert results['depth_cutoff']['status'] == 'no_plan_exhausted'  # goal is reachable at depth 2
results['callback_error'] = plan(broken, {'x':0}, lambda z:False, ['UP'])
assert results['callback_error']['status'] == 'no_plan_exhausted'
results['branch_mutation'] = plan(mutating, {'x':0}, lambda z:z['x']==11, [1,10])
assert results['branch_mutation']['plan'] == [10]  # replay of [10] from x=0 yields x=10, not 11
results['node_limit'] = plan(lambda z,a: {'x':a}, {'x':-1}, lambda z:False, list(range(1000)), max_nodes=1)
assert results['node_limit']['seen'] == 1001
started = time.monotonic()
results['goal_deadline'] = plan(lambda z,a:z, {'x':0}, lambda z:(time.sleep(.03) or True), ['UP'], max_seconds=.001)
results['goal_deadline']['wall_seconds'] = time.monotonic()-started
assert results['goal_deadline']['status'] == 'already_satisfied'
assert results['goal_deadline']['wall_seconds'] > .02
try:
    plan(lambda z,a: {'x':object()}, {'x':0}, lambda z:False, ['UP'])
except TypeError:
    results['serialization_error'] = 'TypeError escapes status contract'
else:
    raise AssertionError('Expected uncaught key serialization error')
print(json.dumps(results, indent=2))
