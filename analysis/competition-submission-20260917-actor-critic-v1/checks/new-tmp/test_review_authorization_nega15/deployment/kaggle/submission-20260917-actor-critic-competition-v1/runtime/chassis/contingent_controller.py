"""Bounded two-decision planning over proposed worlds, never hidden world IDs.

Completion and costs are model estimates, not measured reward or calibrated RHAE.
Within each observable branch the same continuation is used for every world.
"""
import hashlib
import json
import math
import os

from decision_packet import _canonical, _valid_prediction, SUPPORTED_METRICS

ARM_VARIABLE = 'ARC_CONTINGENT_ARM'


def arm():
    """Return the frozen controller arm: 'off' hides the interface, 'on' exposes it.

    The arm is read once per worker process from ARC_CONTINGENT_ARM. It defaults
    to 'on' so every inherited caller keeps its current surface; a matched
    contrast must set both arms explicitly.
    """
    value = os.environ.get(ARM_VARIABLE, 'on').lower()
    if value not in ('off', 'on'):
        raise ValueError(f'{ARM_VARIABLE} must be off or on')
    return value


def enabled():
    """True when the contingent interface is part of this arm's treatment."""
    return arm() == 'on'


def _number(value, low=0, high=10000):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError('Expected a finite bounded numeric estimate')
    return float(value)


def _action(value):
    if value is None:
        return
    if not isinstance(value, str) or len(value) > 32:
        raise ValueError('Expected a primitive action or null')
    parts = value.split()
    if value in ('UP', 'DOWN', 'LEFT', 'RIGHT', 'SPACEBAR', 'USE', 'UNDO', 'RESET'):
        return
    if len(parts) == 3 and parts[0] == 'CLICK' and all(p.isdigit() and 0 <= int(p) <= 63 for p in parts[1:]):
        return
    raise ValueError('Expected a primitive action or null')


def compile_plan(proposal):
    """Maximize declared expected completion, then minimize declared action cost.

    Max four hypotheses, candidates and continuation actions, one observation
    branch point. Remaining actions include the proposed continuation primitive;
    recovery is additional. Equal values favor fewer branches, then input order.
    """
    raw = json.dumps(proposal, allow_nan=False)
    if len(raw) > 24000:
        raise ValueError('Keep the entire proposal within 24000 characters')
    proposal = json.loads(raw)
    if set(proposal) != {'hypotheses', 'candidates'}:
        raise ValueError('Expected hypotheses and candidates')
    weights = proposal['hypotheses']
    if not isinstance(weights, dict) or not 1 <= len(weights) <= 4:
        raise ValueError('Supply one to four weighted hypotheses')
    if any(not isinstance(h, str) or not h or len(h) > 120 for h in weights):
        raise ValueError('Use short hypothesis labels')
    weights = {h: _number(w, 0.000001, 1) for h, w in weights.items()}
    total = sum(weights.values())
    weights = {h: w / total for h, w in weights.items()}
    candidates = proposal['candidates']
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 4:
        raise ValueError('Supply one to four candidates')
    evaluations = []
    ids = set()
    for candidate in candidates:
        if set(candidate) != {'id', 'action', 'metric', 'predictions', 'continuations', 'assumptions'}:
            raise ValueError('Candidate needs id, action, metric, predictions, continuations, assumptions')
        name = candidate['id']
        if not isinstance(name, str) or not name or len(name) > 120 or name in ids:
            raise ValueError('Candidate IDs must be short and unique')
        ids.add(name)
        _action(candidate['action'])
        if candidate['action'] is None:
            raise ValueError('A root candidate must act')
        assumptions = candidate['assumptions']
        if not isinstance(assumptions, list) or not 1 <= len(assumptions) <= 8 or any(
                not isinstance(a, str) or not a or len(a) > 300 for a in assumptions):
            raise ValueError('State one to eight provisional plan assumptions')
        metric, predictions = candidate['metric'], candidate['predictions']
        if metric not in SUPPORTED_METRICS or not isinstance(predictions, dict) or set(predictions) != set(weights):
            raise ValueError('Predict one supported metric for every hypothesis')
        if any(not _valid_prediction(metric, v) for v in predictions.values()):
            raise ValueError('Invalid typed observable prediction')
        options = candidate['continuations']
        if not isinstance(options, list) or not 1 <= len(options) <= 4:
            raise ValueError('Supply one to four continuation actions')
        actions = set()
        for option in options:
            if set(option) != {'action', 'outcomes'}:
                raise ValueError('Continuation needs action and per-hypothesis outcomes')
            _action(option['action'])
            if option['action'] in actions:
                raise ValueError('Duplicate continuation action')
            actions.add(option['action'])
            if not isinstance(option['outcomes'], dict) or set(option['outcomes']) != set(weights):
                raise ValueError('Estimate every continuation in every hypothesis')
            for estimate in option['outcomes'].values():
                if set(estimate) != {'completion', 'remaining_actions', 'recovery_actions'}:
                    raise ValueError('Estimate completion, remaining_actions and recovery_actions')
                _number(estimate['completion'], 0, 1)
                _number(estimate['remaining_actions'], 0 if option['action'] is None else 1)
                _number(estimate['recovery_actions'])
        groups = {}
        for h, value in predictions.items():
            groups.setdefault(_canonical(value), []).append(h)
        branches = []
        for signature, hypotheses in groups.items():
            mass = sum(weights[h] for h in hypotheses)
            scored = []
            for option in options:
                completion = sum(weights[h] * option['outcomes'][h]['completion'] for h in hypotheses) / mass
                cost = sum(weights[h] * (option['outcomes'][h]['remaining_actions'] +
                            option['outcomes'][h]['recovery_actions']) for h in hypotheses) / mass
                scored.append({'action': option['action'], 'completion': completion, 'cost': cost})
            chosen = max(scored, key=lambda x: (x['completion'], -x['cost']))
            branches.append({'observation': json.loads(signature), 'mass': mass,
                             'hypotheses': hypotheses, 'options': scored, **chosen})
        evaluations.append({'id': name, 'action': candidate['action'], 'metric': metric,
                            'assumptions': assumptions, 'branches': branches,
                            'completion': sum(b['mass'] * b['completion'] for b in branches),
                            'cost': 1 + sum(b['mass'] * b['cost'] for b in branches)})
    selected = max(evaluations, key=lambda x: (x['completion'], -x['cost'], -len(x['branches'])))
    return {'schema': 'contingent-plan-v1', 'proposal': proposal, 'weights': weights,
            'selection_rule': 'estimated_completion_then_action_cost',
            'estimate_status': 'provisional_model_claims', 'evaluations': evaluations,
            'selected': selected, 'proposal_sha256': hashlib.sha256(raw.encode()).hexdigest()}


def select_branch(compiled, observation):
    """Selection sees only the observation, never the simulator's sampled world."""
    return next((b for b in compiled['selected']['branches']
                 if _canonical(b['observation']) == _canonical(observation)), None)
