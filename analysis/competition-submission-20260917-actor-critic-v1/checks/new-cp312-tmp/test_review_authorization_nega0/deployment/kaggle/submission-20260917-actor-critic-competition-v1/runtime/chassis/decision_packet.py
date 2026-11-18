"""Compact evidence and prospective questions; no game semantics or progress score."""
import hashlib
import json
from pathlib import Path

from prediction_feedback import rows, write_row


SUPPORTED_METRICS = {
    'grid_changed', 'changed_bbox', 'shape_changed', 'after_shape',
    'changed_cells', 'color_changes', 'state', 'levels_completed_delta',
}
TARGET_METRICS = {
    'any_change': {'grid_changed'},
    'region': {'changed_bbox'},
    'shape': {'shape_changed', 'after_shape'},
    'change_count': {'changed_cells'},
    'color': {'color_changes'},
    'game_state': {'state'},
    'progress': {'levels_completed_delta'},
}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _valid_prediction(metric, value):
    if metric in ('grid_changed', 'shape_changed'):
        return type(value) is bool
    if metric in ('changed_cells', 'levels_completed_delta'):
        return type(value) is int
    if metric == 'changed_bbox':
        return value is None or (isinstance(value, list) and len(value) == 4
                                 and all(type(x) is int for x in value))
    if metric == 'after_shape':
        return (isinstance(value, list) and len(value) == 2
                and all(type(x) is int and x >= 0 for x in value))
    if metric == 'state':
        return isinstance(value, str) and bool(value.strip())
    if metric == 'color_changes':
        return (isinstance(value, list)
                and all(isinstance(item, dict)
                        and set(item) == {'before_color', 'after_color', 'count'}
                        and type(item['before_color']) is int
                        and type(item['after_color']) is int
                        and type(item['count']) is int and item['count'] > 0
                        for item in value))
    return False


def _validate_outcomes(question, hypotheses, outcomes):
    if not isinstance(outcomes, dict) or set(outcomes) != {
            'target', 'metric', 'predictions', 'updates', 'otherwise'}:
        raise ValueError('Use typed outcomes with target, metric, predictions, updates, and otherwise')
    target = outcomes['target']
    if target not in TARGET_METRICS:
        raise ValueError(f'Unsupported experiment target: {target!r}')
    metric = outcomes['metric']
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f'Unsupported experiment metric: {metric!r}')
    if metric not in TARGET_METRICS[target]:
        allowed = ', '.join(sorted(TARGET_METRICS[target]))
        raise ValueError(f'{target} target requires {allowed}')
    predictions, updates = outcomes['predictions'], outcomes['updates']
    expected = set(hypotheses)
    if not isinstance(predictions, dict) or set(predictions) != expected:
        raise ValueError('Prediction keys must exactly match hypotheses')
    if not isinstance(updates, dict) or set(updates) != expected:
        raise ValueError('Update keys must exactly match hypotheses')
    if any(not isinstance(value, str) or not value.strip() for value in updates.values()):
        raise ValueError('Every hypothesis needs a nonempty knowledge update')
    otherwise = outcomes['otherwise']
    if not isinstance(otherwise, str) or not otherwise.strip():
        raise ValueError('Describe the knowledge gained if no prediction matches')
    if any(not _valid_prediction(metric, value) for value in predictions.values()):
        raise ValueError(f'Predictions must use the host type for {metric}')
    if len({_canonical(value) for value in predictions.values()}) != len(predictions):
        raise ValueError('Hypotheses must make distinct predictions')


def _observed_metric(metric, transition, experiment):
    view = transition_view(transition)
    before, after = transition['before'], transition['after']
    if metric == 'grid_changed':
        return view['shape_changed'] or bool(view['changed_cells'])
    if metric in ('changed_bbox', 'shape_changed', 'changed_cells', 'color_changes'):
        return view[metric]
    if metric == 'after_shape':
        grid = after['grid']
        return [len(grid), len(grid[0]) if grid else 0]
    if metric == 'state':
        return after.get('state')
    if metric == 'levels_completed_delta':
        baseline = before.get('levels_completed', experiment.get('levels_completed', 0))
        return after.get('levels_completed', baseline) - baseline
    raise ValueError(f'Unsupported experiment metric: {metric!r}')


def score_experiments(ipc, transitions):
    """Resolve each recorded experiment against its first matching transition once."""
    ipc = Path(ipc)
    experiments = rows(ipc / 'decision_experiments.jsonl')
    existing = rows(ipc / 'experiment_results.jsonl')
    scored_ids = {row['experiment_id'] for row in existing}
    for experiment in experiments:
        if experiment['id'] in scored_ids:
            continue
        transition = next((row for row in transitions
                           if row.get('turn') == experiment['turn']
                           and row.get('step', -1) > experiment['step']
                           and row.get('action') == experiment['action']), None)
        if transition is None:
            continue
        outcomes = experiment['outcomes']
        observed = _observed_metric(outcomes['metric'], transition, experiment)
        matches = [name for name, prediction in outcomes['predictions'].items()
                   if _canonical(prediction) == _canonical(observed)]
        matched = matches[0] if matches else None
        result = {
            'schema': 'decision-experiment-result-v2',
            'experiment_id': experiment['id'],
            'turn': experiment['turn'],
            'experiment_step': experiment['step'],
            'transition_step': transition['step'],
            'action': experiment['action'],
            'metric': outcomes['metric'],
            'observed': observed,
            'status': 'matched' if matched else 'none_of_predictions',
            'matched_prediction_label': matched,
            'authority': 'measured_prediction_only',
            'meaning': 'A matching consequence does not certify the hypothesis or proposed update. Preserve confounders and tested scope.',
            'knowledge': [
                {'hypothesis_label': name,
                 'prediction': outcomes['predictions'][name],
                 'matched_observation': name == matched,
                 'model_proposed_update': outcomes['updates'][name]}
                for name in experiment['hypotheses']
            ],
        }
        if matched is None:
            result['otherwise'] = outcomes['otherwise']
        write_row(ipc / 'experiment_results.jsonl', result)
        existing.append(result)
        scored_ids.add(experiment['id'])
    return existing


def transition_view(row, include_grids=False):
    """Measured pixel evidence; no inferred object identities or dynamics."""
    before, after = row['before']['grid'], row['after']['grid']
    same = len(before) == len(after) and all(len(a)==len(b) for a,b in zip(before,after))
    changes = [[r,c,x,y] for r,(a,b) in enumerate(zip(before,after))
               for c,(x,y) in enumerate(zip(a,b)) if x != y] if same else []
    counts = {}
    for _,_,old,new in changes:
        counts[old,new] = counts.get((old,new),0) + 1
    digest = lambda grid: hashlib.sha256(json.dumps(grid,separators=(',',':')).encode()).hexdigest()
    result = dict(kind='measured_transition', step=row['step'], turn=row.get('turn'),
        action=row['action'], before_sha256=digest(before), after_sha256=digest(after),
        shape_changed=not same, changed_cells=len(changes) if same else None,
        changed_bbox=([min(x[0] for x in changes), min(x[1] for x in changes),
                       max(x[0] for x in changes), max(x[1] for x in changes)] if changes else None),
        changes=changes[:64], changes_omitted=max(0,len(changes)-64),
        change_columns=['row','column','before_color','after_color'],
        color_changes=[dict(before_color=a,after_color=b,count=n) for (a,b),n in sorted(counts.items())],
        meaning='Measured pixels only. Object identity, cause, and objective remain hypotheses.')
    if include_grids:
        result['before_grid_rows'] = [''.join(format(v,'x') for v in r) for r in before]
        result['after_grid_rows'] = [''.join(format(v,'x') for v in r) for r in after]
    return result


def record_experiment(ipc, state, question, hypotheses, action, outcomes):
    if not isinstance(question, str) or not question.strip():
        raise ValueError('Name the unresolved question')
    if not isinstance(hypotheses, list) or not 2 <= len(hypotheses) <= 3:
        raise ValueError('Supply two or three competing hypotheses')
    if any(not isinstance(h, str) or not h.strip() for h in hypotheses):
        raise ValueError('Hypotheses must be nonempty text')
    if not isinstance(action, str) or not action.strip():
        raise ValueError('Name the proposed action or bounded test')
    if len(set(hypotheses)) != len(hypotheses):
        raise ValueError('Hypotheses must be distinct')
    _validate_outcomes(question, hypotheses, outcomes)
    if not state or 'turn' not in state or 'step' not in state:
        raise ValueError('No current observation')
    record = dict(schema='decision-experiment-v2', turn=state['turn'], step=state['step'],
                  levels_completed=state.get('levels_completed', 0), question=question,
                  hypotheses=hypotheses, action=action, outcomes=outcomes)
    raw = json.dumps(record, allow_nan=False)
    if len(raw) > 4000:
        raise ValueError('Keep the changed decision record within 4000 characters')
    record['id'] = hashlib.sha256(raw.encode()).hexdigest()
    write_row(Path(ipc) / 'decision_experiments.jsonl', record)
    return record


def packet(state, transitions, feedback, observations, helpers, ipc):
    recent = []
    for row in transitions[-3:]:
        before, after = row['before']['grid'], row['after']['grid']
        same_shape = len(before) == len(after) and all(len(a) == len(b) for a, b in zip(before, after))
        changed = [(r, c) for r, (a, b) in enumerate(zip(before, after))
                   for c, (x, y) in enumerate(zip(a, b)) if x != y] if same_shape else []
        recent.append(dict(step=row['step'], action=row['action'],
                           changed_cells=len(changed) if same_shape else None,
                           shape_changed=not same_shape,
                           changed_bbox=([min(r for r,c in changed), min(c for r,c in changed),
                                          max(r for r,c in changed), max(c for r,c in changed)] if changed else None),
                           state=row['after'].get('state'),
                           levels_completed=row['after'].get('levels_completed')))
    experiments = rows(Path(ipc) / 'decision_experiments.jsonl')
    results = score_experiments(ipc, transitions)
    experiment = experiments[-1] if experiments else None
    experiment_result = next((row for row in reversed(results)
                              if experiment and row['experiment_id'] == experiment['id']), None)
    return dict(schema='decision-packet-v1',
                state={k:v for k,v in state.items() if k in ('turn','step','level','state','levels_completed','available_actions','note')},
                recent_transitions=recent, prediction_feedback=feedback,
                last_transition=transition_view(transitions[-1]) if transitions else None,
                known_from_here=observations.get('known_from_here', {}),
                helper_versions={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(Path(helpers).glob('*.py'))},
                experiment=experiment,
                experiment_result=experiment_result,
                experiment_from_current_observation=bool(experiment and all(experiment.get(k)==state.get(k) for k in ('turn','step','levels_completed'))),
                limits='Changed pixels and encoder matches are not progress. Full grids remain at self.latest_state(); history at self.trajectory().')
