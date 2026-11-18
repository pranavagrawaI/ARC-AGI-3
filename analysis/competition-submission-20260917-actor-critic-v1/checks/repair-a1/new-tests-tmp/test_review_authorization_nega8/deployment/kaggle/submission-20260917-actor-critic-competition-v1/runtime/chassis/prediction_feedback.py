"""Prospective consistency checks for the Actor's own executable conjecture.

No game semantics live here. The frozen encoder is a measurement hypothesis,
not ground truth; a matching latent state never establishes model completeness.
"""
from __future__ import annotations

import hashlib
import json
import signal
import threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np


class HelperTimeoutError(TimeoutError):
    pass


@contextmanager
def computation_budget(seconds=1.0):
    """Bound synchronous helper execution without borrowing another alarm.

    This Linux/main-thread guard interrupts Python loops. A native extension
    which defers Python signal handling can still exceed this deadline; the
    existing worker timeout remains the outer containment boundary.
    """
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('prediction evaluation needs the main thread timer')
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer != (0.0, 0.0):
        raise RuntimeError('prediction evaluation unavailable: another alarm is active')
    previous_handler = signal.getsignal(signal.SIGALRM)
    def expired(_signum, _frame):
        raise HelperTimeoutError(f'executable model exceeded {seconds:g}s computation budget')
    signal.signal(signal.SIGALRM, expired)
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def rows(path):
    if not path.exists():
        return []
    text = path.read_text()
    lines = text.splitlines()
    result = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            # The host may still be appending its final transition. Only an
            # unterminated tail is incomplete; corrupt durable rows stay errors.
            if index == len(lines) - 1 and not text.endswith("\n"):
                break
            raise
    return result


def write_row(path, row):
    with path.open('a') as stream:
        stream.write(json.dumps(row, allow_nan=False) + '\n')


def plain(value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    encoded = json.dumps(value, default=convert, allow_nan=False)
    if len(encoded) > 16000:
        raise ValueError('latent exceeds 16000 JSON characters; use a compact representation')
    value = json.loads(encoded)
    if not isinstance(value, dict) or not value:
        raise ValueError('encode/predict must return a nonempty JSON object')
    return value


def select_model(helpers):
    """Select and snapshot source without executing it; one owner for model identity."""
    # Prefer the documented names. Other helpers are accepted only when
    # exactly one contains both entry points; never silently switch models.
    files = list(helpers.glob('*.py'))
    preferred = [helpers / name for name in ('world_model.py', 'worldmodel.py')]
    candidates = [p for p in preferred if p.exists()]
    if not candidates:
        candidates = [p for p in files if 'def encode(' in p.read_text() and 'def predict(' in p.read_text()]
    if not candidates:
        return 'no executable model yet; one exploratory action is allowed'
    if len(candidates) > 1 and not any(p.name == 'world_model.py' for p in candidates):
        return 'ambiguous model helpers; use world_model.py'
    path = candidates[0]
    source = path.read_text()
    return dict(model=path.name, source=source,
                model_sha256=hashlib.sha256(source.encode()).hexdigest())


class PredictionFeedback:
    def __init__(self, ipc: Path, helpers: Path, validate_source):
        self.ipc, self.helpers, self.validate_source = ipc, helpers, validate_source
        self.commits = ipc / 'model_predictions.jsonl'
        self.results = ipc / 'model_feedback.jsonl'

    def _module(self, source):
        error = self.validate_source(source)
        if error:
            raise ValueError(error)
        namespace = {'__name__': 'frozen_conjecture'}
        exec(compile(source, '<frozen-conjecture>', 'exec'), namespace)
        if not all(callable(namespace.get(name)) for name in ('encode', 'predict')):
            raise ValueError('helper requires encode(grid) and predict(z, action)')
        return namespace

    def prepare(self, state, actions):
        record = {'schema': 'q6-model-prediction-v1', 'turn': state['turn'],
                  'step': state['step'], 'levels_completed': state.get('levels_completed', 0),
                  'actions': list(actions), 'status': 'unavailable'}
        selected = select_model(self.helpers)
        if isinstance(selected, str):
            return {**record, 'reason': selected}
        record.update(selected)
        source = selected['source']
        try:
            with computation_budget():
                module = self._module(source)
                grid = state.get('grid')
                if grid is None:
                    grid = [[int(cell, 16) for cell in row] for row in state['grid_rows']]
                elif isinstance(grid, list) and grid and isinstance(grid[0], str):
                    grid = [[int(cell, 16) for cell in row] for row in grid]
                latent = plain(module['encode'](np.array(grid)))
                record['before'] = latent
                predicted = []
                for action in actions:
                    # Isolate mutation of the input latent from the saved evidence.
                    latent = plain(module['predict'](json.loads(json.dumps(latent)), action))
                    predicted.append(latent)
                record.update(status='committed', predicted=predicted)
        except (Exception, SystemExit, KeyboardInterrupt) as exc:
            record['reason'] = f'{type(exc).__name__}: {exc}'
        return record

    def preflight(self, state, actions):
        """Exercise exact preparation without publishing a prediction or action."""
        if not isinstance(actions, list) or not 1 <= len(actions) <= 20 or any(
                not isinstance(a, str) or not a.strip() for a in actions):
            raise ValueError('Supply one to twenty action strings')
        if not state:
            return dict(kind='helper_preflight', status='unavailable', reason='no current observation',
                        meaning='No prediction or action committed.')
        record = self.prepare(state, actions)
        grid = state.get('grid')
        if grid is None and state.get('grid_rows') is not None:
            grid = [[int(cell, 16) for cell in row] for row in state['grid_rows']]
        elif isinstance(grid, list) and grid and isinstance(grid[0], str):
            grid = [[int(cell, 16) for cell in row] for row in grid]
        array = np.asarray(grid, dtype=int) if grid is not None else None
        input_contract = {
            'encode_input': 'numpy.ndarray[int] (detached)',
            'shape': list(array.shape) if array is not None else None,
            'dtype': str(array.dtype) if array is not None else None,
            'predict_input': 'JSON-normalized object (tuples become lists)',
            'latent_limit_chars': 16000,
        }
        return dict(kind='helper_preflight', status='ready' if record['status']=='committed' else 'unavailable',
            turn=record['turn'], step=record['step'], actions=list(actions),
            model=record.get('model'), model_sha256=record.get('model_sha256'),
            reason=record.get('reason'), input_contract=input_contract,
            meaning='Execution compatibility only; not evidence of correct dynamics. No prediction or action committed.')

    def commit(self, record):
        write_row(self.commits, record)

    def observe(self):
        commits = rows(self.commits)
        if not commits:
            return {'status': 'unmeasured', 'batch_allowed': False}
        record = commits[-1]
        transitions = [r for r in rows(self.ipc / 'transitions.jsonl')
                       if record['step'] < r['step'] <= record['step'] + len(record['actions'])
                       and r['turn'] == record['turn']]
        result = {'schema': 'q6-model-feedback-v1', 'turn': record['turn'],
                  'start_step': record['step'], 'model_sha256': record.get('model_sha256'),
                  'model': record.get('model'), 'status': 'pending', 'checked': 0,
                  'differences': [], 'batch_allowed': False,
                  'meaning': 'Consistency under your frozen encoder, not proof of game rules. '
                             'Repair contradicted predictions; test unresolved explanations with one action.'}
        if record['status'] != 'committed':
            result.update(status='unavailable', reason=record.get('reason'))
        elif transitions:
            try:
                with computation_budget():
                    module = self._module(record['source'])
                    for i, transition in enumerate(transitions):
                        if transition['step'] != record['step'] + i + 1 or transition['action'] != record['actions'][i]:
                            raise ValueError('transition does not match committed action prefix')
                        if transition['after'].get('levels_completed', 0) != record['levels_completed']:
                            result['boundary'] = 'level changed; revalidate the model on this level'
                            break
                        actual = plain(module['encode'](np.array(transition['after']['grid'])))
                        expected = record['predicted'][i]
                        diffs = {key: {'predicted': expected.get(key), 'observed': actual.get(key),
                                       'predicted_present': key in expected, 'observed_present': key in actual}
                                 for key in sorted(set(expected) | set(actual))
                                 if key not in expected or key not in actual or expected[key] != actual[key]}
                        result['checked'] += 1
                        result['differences'].append({'step': transition['step'], 'action': transition['action'],
                                                       'fields': diffs})
                    mismatch = any(r['fields'] for r in result['differences'])
                    result['status'] = 'mismatch' if mismatch else ('matched' if result['checked'] else 'unmeasured')
                    result['batch_allowed'] = result['status'] == 'matched' and 'boundary' not in result
            except (Exception, SystemExit, KeyboardInterrupt) as exc:
                result.update(status='unavailable', reason=f'{type(exc).__name__}: {exc}', batch_allowed=False)
        previous = rows(self.results)
        if not previous or previous[-1] != result:
            write_row(self.results, result)
        return result

    def batch_error(self, proposal, feedback):
        if (proposal['status'] == 'committed' and feedback.get('batch_allowed')
                and proposal.get('model_sha256') == feedback.get('model_sha256')):
            return None
        return ('Batching needs a successful prospective check of this helper version. '
                'Inspect self.prediction_feedback(), repair or narrow the model, and submit ONE action. '
                'An unresolved model never blocks a single exploratory action.')
