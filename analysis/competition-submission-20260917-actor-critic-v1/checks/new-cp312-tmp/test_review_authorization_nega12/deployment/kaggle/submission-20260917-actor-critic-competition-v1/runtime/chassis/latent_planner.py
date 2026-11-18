"""Bounded conditional search, never world evidence or action authorization.

Linux fork containment deliberately fails closed outside the main thread. The
parent owns the wall deadline and reaps children; callbacks never run in it.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import signal
import tempfile
import threading
import time

import numpy as np
from prediction_feedback import PredictionFeedback, plain, select_model, computation_budget, HelperTimeoutError

CEILINGS = dict(wall_seconds=5.0, callback_seconds=0.25, generated_states=5000,
                frontier=1000, depth=12, action_count=64)


class InvalidRequest(ValueError):
    pass


class Limited(Exception):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def grid_of(state):
    grid = state.get('grid', state.get('grid_rows'))
    if grid and isinstance(grid[0], str):
        grid = [[int(c, 16) for c in row] for row in grid]
    return grid


def identity(state, selected):
    return dict(turn=state.get('turn'), step=state.get('step'),
                level=state.get('level'), levels_completed=state.get('levels_completed', 0),
                grid_sha256=hashlib.sha256(canonical(grid_of(state)).encode()).hexdigest(),
                model_sha256=selected['model_sha256'], model=selected['model'])


def validate_target(target, depth=0):
    if depth > 8 or not isinstance(target, dict):
        raise InvalidRequest('target must be a bounded JSON predicate')
    if set(target) in ({'all'}, {'any'}):
        values = next(iter(target.values()))
        if not isinstance(values, list) or not 1 <= len(values) <= 16:
            raise InvalidRequest('all/any needs 1..16 predicates')
        for item in values:
            validate_target(item, depth + 1)
    elif (set(target) == {'path', 'op', 'value'} and
          target['op'] in ('eq', 'lt', 'le', 'gt', 'ge') and
          isinstance(target['path'], list) and 1 <= len(target['path']) <= 16 and
          all(type(k) in (str, int) for k in target['path'])):
        return
    else:
        raise InvalidRequest('use path/op/value or all/any composition')


def target_holds(target, z):
    if 'all' in target or 'any' in target:
        op = 'all' if 'all' in target else 'any'
        # Evaluate all leaves: missing fields never disappear via short circuit.
        values = [target_holds(t, z) for t in target[op]]
        return all(values) if op == 'all' else any(values)
    value = z
    try:
        for k in target['path']:
            if isinstance(value, dict) and type(k) is str:
                value = value[k]
            elif isinstance(value, list) and type(k) is int and 0 <= k < len(value):
                value = value[k]
            else:
                raise KeyError(k)
        other, op = target['value'], target['op']
        if op == 'eq':
            return canonical(value) == canonical(other)
        if type(value) not in (int, float) or type(other) not in (int, float):
            raise TypeError('ordering requires numeric fields and values')
        return {'lt': lambda: value < other, 'le': lambda: value <= other,
                'gt': lambda: value > other, 'ge': lambda: value >= other}[op]()
    except (KeyError, IndexError, TypeError) as exc:
        raise InvalidRequest(f'target field/type error: {exc}') from exc


def validate_actions(actions, state, cap):
    if actions is None:
        actions = [a for a in state.get('available_actions', []) if a != 'CLICK']
    if not isinstance(actions, list) or not 1 <= len(actions) <= cap:
        raise InvalidRequest('supply 1..action_count candidates; CLICK needs explicit targets')
    available = state.get('available_actions', [])
    for a in actions:
        if not isinstance(a, str):
            raise InvalidRequest('actions must be canonical strings')
        parts = a.split()
        if not parts or parts[0] not in available:
            raise InvalidRequest('action is not currently available')
        if parts[0] == 'CLICK':
            try:
                valid = len(parts) == 3 and all(str(int(x)) == x and 0 <= int(x) < 64 for x in parts[1:])
            except ValueError:
                valid = False
            if not valid:
                raise InvalidRequest('use CLICK x y with integer coordinates 0..63')
        elif len(parts) != 1:
            raise InvalidRequest('non-CLICK actions take no arguments')
        if a != ' '.join(parts):
            raise InvalidRequest('action must be canonical')
    return list(dict.fromkeys(actions))


def _search(state, helpers, validate_source, target, actions, limits):
    result = dict(status='model_error', snapshot=None, initial_latent=None,
                  expanded_nodes=0, generated_states=0, seen_states=0,
                  errors=[], error_count=0, complete=False, limits=limits)
    stage = 'selection'
    def invoke(fn, *args):
        with computation_budget(limits['callback_seconds']):
            return fn(*args)
    try:
        selected = select_model(helpers)
        if isinstance(selected, str):
            result.update(status='unavailable', reason=selected)
            return result
        result['snapshot'] = identity(state, selected)
        result['model_source'] = selected['source']
        result['target'], result['actions'] = target, actions
        feedback = PredictionFeedback(Path('.'), helpers, validate_source)
        stage = 'load'
        module = invoke(feedback._module, selected['source'])
        stage = 'encode'
        z0 = invoke(lambda: plain(module['encode'](np.array(grid_of(state)))))
        result['initial_latent'] = z0
        stage = 'target'
        if invoke(target_holds, target, z0):
            result.update(status='already_satisfied', plan=[], predicted_latents=[])
            return result
        # Parent pointers avoid duplicating paths/frontiers per node.
        nodes = [(z0, None, None, 0)]
        seen, q = {canonical(z0)}, deque([0])
        result['seen_states'] = 1
        depth_cutoff = False
        while q:
            index = q.popleft()
            z, _, _, depth = nodes[index]
            if depth >= limits['depth']:
                depth_cutoff = True
                continue
            result['expanded_nodes'] += 1
            for action in actions:
                if result['generated_states'] >= limits['generated_states']:
                    raise Limited('generated_states')
                result['generated_states'] += 1
                stage = 'predict'
                nz = invoke(lambda: plain(module['predict'](json.loads(canonical(z)), action)))
                key = canonical(nz)
                if key in seen:
                    continue
                seen.add(key)
                result['seen_states'] = len(seen)
                stage = 'target'
                hit = invoke(target_holds, target, nz)
                if hit:
                    plan, latents = [action], [nz]
                    while nodes[index][1] is not None:
                        zz, parent, aa, _ = nodes[index]
                        plan.append(aa); latents.append(zz); index = parent
                    plan.reverse(); latents.reverse()
                    stage = 'replay_load'
                    replay = invoke(feedback._module, selected['source'])
                    stage = 'replay_encode'
                    rz = invoke(lambda: plain(replay['encode'](np.array(grid_of(state)))))
                    if canonical(rz) != canonical(z0):
                        raise ValueError('fresh encoder disagrees with initial latent')
                    for aa, expected in zip(plan, latents):
                        stage = 'replay_predict'
                        rz = invoke(lambda: plain(replay['predict'](json.loads(canonical(rz)), aa)))
                        if canonical(rz) != canonical(expected):
                            raise ValueError('fresh replay disagrees with searched path')
                    stage = 'replay_target'
                    if not invoke(target_holds, target, rz):
                        raise ValueError('replayed path does not satisfy target')
                    result.update(status='path_found', plan=plan, predicted_latents=latents,
                                  replay='matched')
                    return result
                if len(q) >= limits['frontier']:
                    raise Limited('frontier')
                nodes.append((nz, index, action, depth + 1))
                q.append(len(nodes)-1)
        result.update(status='limited' if depth_cutoff else 'model_exhausted',
                      complete=not depth_cutoff)
        if depth_cutoff:
            result['reason'] = 'depth'
    except (Limited, HelperTimeoutError) as exc:
        result.update(status='limited', reason='callback_time' if isinstance(exc, HelperTimeoutError) else str(exc), stage=stage)
    except InvalidRequest as exc:
        result.update(status='invalid_request', reason=str(exc)[:500], stage=stage)
    except BaseException as exc:
        result.update(status='model_error', stage=stage,
                      errors=[f'{type(exc).__name__}: {exc}'[:500]], error_count=1)
    return result


def _child(destination, state, helpers, validate_source, target, actions, limits):
    signal.setitimer(signal.ITIMER_REAL, 0)
    # Model stdout must not masquerade as the tool result or fill a pipe.
    with open(os.devnull, 'w') as sink:
        import contextlib
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            result = _search(state, helpers, validate_source, target, actions, limits)
    Path(destination).write_text(canonical(result))


def run(state, helpers, validate_source, target, actions=None, limits=None):
    started = time.monotonic()
    base = dict(status='unavailable', snapshot=None, initial_latent=None, expanded_nodes=0,
                generated_states=0, seen_states=0, complete=False, errors=[], error_count=0)
    proc = None
    try:
        supplied = {} if limits is None else limits
        if not isinstance(supplied, dict) or set(supplied) - set(CEILINGS):
            raise InvalidRequest('unknown limits; use ' + ', '.join(CEILINGS))
        effective = {**CEILINGS, **supplied}
        for name, value in effective.items():
            expected = (int, float) if name.endswith('seconds') else (int,)
            if type(value) not in expected or not 0 < value <= CEILINGS[name]:
                raise InvalidRequest(f'{name} must be positive and at most {CEILINGS[name]}')
        base['limits'] = effective
        # Bound declarative input before traversal; deep/recursive objects fail closed.
        raw = canonical(target)
        if len(raw) > 8000:
            raise InvalidRequest('target exceeds 8000 characters')
        target = json.loads(raw)
        validate_target(target)
        if not state:
            raise InvalidRequest('no current state')
        actions = validate_actions(actions, state, effective['action_count'])
        if os.name != 'posix' or threading.current_thread() is not threading.main_thread():
            base['reason'] = 'fork containment requires a POSIX main thread'
            return base
        with tempfile.TemporaryDirectory(prefix='latent-search-') as directory:
            destination = str(Path(directory)/'result.json')
            proc = multiprocessing.get_context('fork').Process(target=_child,
                args=(destination, state, helpers, validate_source, target, actions, effective))
            proc.start()
            pid = proc.pid
            proc.join(max(0, effective['wall_seconds'] - (time.monotonic() - started)))
            if proc.is_alive():
                proc.kill(); proc.join()
                base.update(status='limited', reason='wall_time', child_pid=pid, child_reaped=True,
                            expanded_nodes=None, generated_states=None, seen_states=None)
            elif proc.exitcode != 0 or not Path(destination).exists():
                base.update(status='model_error', reason='search child failed', child_pid=pid, child_reaped=True)
            else:
                base = json.loads(Path(destination).read_text())
                base.update(child_pid=pid, child_reaped=True)
            # A late successful child may not turn a missed deadline into a plan.
            if time.monotonic() - started > effective['wall_seconds']:
                base.update(status='limited', reason='wall_time', complete=False)
                base.pop('plan', None); base.pop('predicted_latents', None)
        return base
    except (InvalidRequest, TypeError, ValueError, RecursionError) as exc:
        base.update(status='invalid_request', reason=str(exc)[:500])
        return base
    except Exception as exc:
        base.update(status='unavailable', reason=f'{type(exc).__name__}: {exc}'[:500])
        return base
    finally:
        if proc is not None:
            if proc.is_alive():
                proc.kill(); proc.join()
            proc.close()
        base['elapsed_seconds'] = time.monotonic() - started
        base['meaning'] = 'Conditional on this deterministic Markov model and action set; no world impossibility, objective proof, or batch permission.'
