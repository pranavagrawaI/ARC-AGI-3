"""Compact evidence and prospective questions; no game semantics or progress score."""
import hashlib
import json
from pathlib import Path

from prediction_feedback import rows, write_row


def record_experiment(ipc, state, question, hypotheses, action, outcomes):
    if not isinstance(question, str) or not question.strip():
        raise ValueError('Name the unresolved question')
    if not isinstance(hypotheses, list) or not 1 <= len(hypotheses) <= 3:
        raise ValueError('Supply one to three provisional hypotheses; do not invent alternatives')
    if any(not isinstance(h, str) or not h.strip() for h in hypotheses):
        raise ValueError('Hypotheses must be nonempty text')
    if not isinstance(action, str) or not action.strip():
        raise ValueError('Name the proposed action or bounded test')
    if not isinstance(outcomes, dict) or len(outcomes) < 2:
        raise ValueError('Map at least two observable outcomes (including inconclusive) to belief updates')
    if any(not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not v.strip()
           for k, v in outcomes.items()):
        raise ValueError('Outcome and update descriptions must be nonempty strings')
    if not state or 'turn' not in state or 'step' not in state:
        raise ValueError('No current observation')
    record = dict(schema='decision-experiment-v1', turn=state['turn'], step=state['step'],
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
    experiment = experiments[-1] if experiments else None
    return dict(schema='decision-packet-v1',
                state={k:v for k,v in state.items() if k in ('turn','step','level','state','levels_completed','available_actions','note')},
                recent_transitions=recent, prediction_feedback=feedback,
                known_from_here=observations.get('known_from_here', {}),
                helper_versions={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(Path(helpers).glob('*.py'))},
                experiment=experiment,
                experiment_from_current_observation=bool(experiment and all(experiment.get(k)==state.get(k) for k in ('turn','step','levels_completed'))),
                limits='Changed pixels and encoder matches are not progress. Full grids remain at self.latest_state(); history at self.trajectory().')
