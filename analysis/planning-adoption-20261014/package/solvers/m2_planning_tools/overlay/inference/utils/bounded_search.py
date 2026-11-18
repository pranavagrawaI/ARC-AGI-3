"""Unit-cost graph search with model-owned rules and explicit stopping reasons.

Callbacks must be pure and state_key must identify equivalent future behavior.
The optional heuristic supplies priority guidance; arbitrary heuristics do not
guarantee shortest routes. Callbacks are cooperatively timed; the enclosing
sandbox supplies the hard deadline for a callback that never returns.
"""
import heapq as _search_heapq
import math as _search_math
import time as _search_time


def _search_caps(max_depth, max_nodes, max_expanded, max_seconds):
    if max_depth is not None and (type(max_depth) is not int or max_depth < 0):
        raise ValueError('max_depth must be a nonnegative integer or None')
    for name, value in (('max_nodes', max_nodes), ('max_expanded', max_expanded)):
        if type(value) is not int or value < 1:
            raise ValueError(name + ' must be a positive integer')
    if isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float)):
        raise ValueError('max_seconds must be finite and nonnegative')
    if not _search_math.isfinite(max_seconds) or max_seconds < 0:
        raise ValueError('max_seconds must be finite and nonnegative')


class _SearchFrontier:
    """Own stable predecessor records, stale entries, and best known depths."""
    def __init__(self):
        self.nodes = []
        self.best = {}
        self.heap = []

    def add(self, state, key, depth, parent, action, heuristic):
        index = len(self.nodes)
        self.nodes.append((state, key, depth, parent, action))
        self.best[key] = index
        _search_heapq.heappush(self.heap, (depth + heuristic, heuristic, index))
        return index

    def pop(self):
        while self.heap:
            _, _, index = _search_heapq.heappop(self.heap)
            state, key, depth, _, _ = self.nodes[index]
            if self.best[key] == index:
                return index, state, key, depth
        return None

    def improves(self, key, depth):
        prior = self.best.get(key)
        return prior is None or depth < self.nodes[prior][2]

    def path(self, index):
        actions = []
        while self.nodes[index][3] is not None:
            _, _, _, parent, action = self.nodes[index]
            actions.append(action)
            index = parent
        actions.reverse()
        return actions


def bounded_search(initial_state, successors, *, state_key, is_goal, heuristic=None,
                   max_depth=None, max_nodes=100000, max_expanded=50000,
                   max_seconds=10.0):
    """Search without executing actions; return a result dictionary.

    successors(state) yields (action, next_state) pairs; next_state=None rejects
    that edge. state_key(state) must return a stable hashable key; is_goal(state)
    returns bool. heuristic(state), if supplied, returns a finite nonnegative
    number. The zero heuristic is breadth-first search for unit-cost edges.
    max_nodes bounds stored predecessor records, including reopened states.
    Limits are checked before and after callbacks and between yielded edges.
    FOUND certifies a path in the supplied model only, not in the real game.
    """
    _search_caps(max_depth, max_nodes, max_expanded, max_seconds)
    for name, callback in (('successors', successors), ('state_key', state_key), ('is_goal', is_goal)):
        if not callable(callback):
            raise TypeError(name + ' must be callable')
    if heuristic is not None and not callable(heuristic):
        raise TypeError('heuristic must be callable or None')
    started = _search_time.monotonic()
    frontier = _SearchFrontier()
    expanded, generated, rejected = 0, 0, 0
    depth_cut = set()
    phase = 'initial'

    def timed_out():
        return _search_time.monotonic() - started >= max_seconds

    def priority(state):
        value = 0 if heuristic is None else heuristic(state)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError('heuristic must return a finite nonnegative number')
        if not _search_math.isfinite(value) or value < 0:
            raise ValueError('heuristic must return a finite nonnegative number')
        return value

    def finish(status, *, reason=None, index=None, error=None):
        result = {'status': status, 'reason': reason, 'path': None, 'state': None,
                  'depth': None, 'expanded': expanded, 'generated': generated,
                  'rejected': rejected, 'discovered': len(frontier.best),
                  'stored_nodes': len(frontier.nodes), 'depth_frontier_states': len(depth_cut),
                  'seconds': _search_time.monotonic() - started}
        if index is not None:
            result.update(path=frontier.path(index), state=frontier.nodes[index][0],
                          depth=frontier.nodes[index][2])
        if error is not None:
            result.update(callback=phase, error=type(error).__name__ + ': ' + str(error))
        return result

    try:
        if timed_out():
            return finish('RESOURCE_LIMIT', reason='TIME_LIMIT')
        phase = 'state_key'
        key = state_key(initial_state)
        hash(key)
        if timed_out():
            return finish('RESOURCE_LIMIT', reason='TIME_LIMIT')
        phase = 'heuristic'
        h = priority(initial_state)
        if timed_out():
            return finish('RESOURCE_LIMIT', reason='TIME_LIMIT')
        frontier.add(initial_state, key, 0, None, None, h)
        while True:
            if timed_out():
                return finish('RESOURCE_LIMIT', reason='TIME_LIMIT')
            node = frontier.pop()
            if node is None:
                return finish('HORIZON_EXHAUSTED' if depth_cut else 'GRAPH_EXHAUSTED')
            index, state, key, depth = node
            phase = 'is_goal'
            goal = is_goal(state)
            if type(goal) is not bool:
                raise ValueError('is_goal must return bool')
            if timed_out():
                return finish('RESOURCE_LIMIT', reason='TIME_LIMIT')
            if goal:
                return finish('FOUND', index=index)
            if max_depth is not None and depth >= max_depth:
                depth_cut.add(key)
                continue
            if expanded >= max_expanded:
                return finish('RESOURCE_LIMIT', reason='EXPANSION_LIMIT')
            expanded += 1
            phase = 'successors'
            for action, successor in successors(state):
                if timed_out():
                    return finish('RESOURCE_LIMIT', reason='TIME_LIMIT')
                generated += 1
                if successor is None:
                    rejected += 1
                    continue
                phase = 'state_key'
                successor_key = state_key(successor)
                hash(successor_key)
                if timed_out():
                    return finish('RESOURCE_LIMIT', reason='TIME_LIMIT')
                if not frontier.improves(successor_key, depth + 1):
                    phase = 'successors'
                    continue
                if len(frontier.nodes) >= max_nodes:
                    return finish('RESOURCE_LIMIT', reason='NODE_LIMIT')
                phase = 'heuristic'
                h = priority(successor)
                if timed_out():
                    return finish('RESOURCE_LIMIT', reason='TIME_LIMIT')
                frontier.add(successor, successor_key, depth + 1, index, action, h)
                # Reopening at a shorter depth retires an older depth cutoff.
                depth_cut.discard(successor_key)
                phase = 'successors'
        # The surrounding sandbox remains responsible for killing hung callbacks.
    except Exception as error:
        return finish('CALLBACK_ERROR', error=error)
