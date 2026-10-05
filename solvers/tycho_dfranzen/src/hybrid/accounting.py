"""Reconcile completed model usage after the worker and its writers stop."""
from __future__ import annotations

import json
from pathlib import Path


def interrupted_builder_calls(session_root: Path) -> int:
    """Unreturned provider responses are unknown costs, not completed-call totals."""
    path = session_root / 'builder-lifecycle.jsonl'
    if not path.exists():
        return 0
    records = path.read_bytes().split(b'\n')
    unknown = int(bool(records.pop()))
    return unknown + sum(json.loads(raw)['phase'] == 'provider_interrupted' for raw in records)


def final_usage(session_root: Path, reported: dict | None) -> tuple[dict, dict]:
    """Prefer durable completed responses to the last, possibly stale IPC snapshot.

    An interrupted final JSONL write is not a completed record. Its usage stays
    unresolved. Malformed committed records or a trace behind the IPC snapshot
    are integrity failures, rather than permission to discard known computation.
    """
    snapshot = dict(reported or {})
    interrupted = interrupted_builder_calls(session_root)
    path = session_root / 'llm-trace.jsonl'
    if not path.exists():
        return snapshot, {'source': 'worker_ipc_snapshot', 'incomplete_trace_tail': False,
                          'unknown_usage_calls': 0, 'unreported_completed_calls': 0,
                          'unreported_completion_tokens': 0,
                          'interrupted_builder_calls_unknown_usage': interrupted}
    data = path.read_bytes()
    records = data.split(b'\n')
    tail_incomplete = bool(records.pop())
    totals = {'calls': 0, 'completion_tokens': 0, 'prompt_tokens': 0, 'cached_tokens': 0,
              'roles': {}}
    unknown_usage = 0
    for raw in records:
        record = json.loads(raw)
        if record['call_index'] != totals['calls'] + 1:
            raise ValueError('completed-response trace call sequence mismatch')
        source = record['source']
        if not isinstance(source, str) or not source:
            raise ValueError('completed-response trace role missing')
        totals['calls'] += 1
        role = totals['roles'].setdefault(source, {'calls': 0, 'completion_tokens': 0})
        role['calls'] += 1
        unknown_usage += any(record[key] is None for key in ('completion_tokens', 'prompt_tokens', 'cached_tokens'))
        for key in ('completion_tokens', 'prompt_tokens', 'cached_tokens'):
            value = record[key]
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError('invalid completed-response token usage')
            totals[key] += value or 0
        role['completion_tokens'] += record['completion_tokens'] or 0
    if any(totals[key] < snapshot.get(key, 0) for key in ('calls', 'completion_tokens')):
        raise ValueError('completed-response trace is behind reported usage')
    return {**snapshot, **totals}, {
        'source': 'retired_worker_llm_trace', 'incomplete_trace_tail': tail_incomplete,
        'unknown_usage_calls': unknown_usage,
        'interrupted_builder_calls_unknown_usage': interrupted,
        'unreported_completed_calls': totals['calls'] - snapshot.get('calls', 0),
        'unreported_completion_tokens': totals['completion_tokens'] - snapshot.get('completion_tokens', 0),
    }
