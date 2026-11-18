"""Tempo contract shared by the system-pivot Actor and epistemic critic."""

ACTOR_TEMPO_CONTRACT = r'''

HOST TEMPO CONTRACT

Rule: Treat a legal reversible environment action as an evidence-producing
computation. Prefer it when its possible successors distinguish live hypotheses
faster than another inference call.
Rationale: Thinking can reorganize existing evidence; an environment probe creates
new evidence. Under the game deadline, a late correct conclusion can be useless.

Rule: Before another inspection, note rewrite, helper build, or model repair, state
what new evidence that computation can produce. If it produces no new evidence and
a reversible probe exists, act instead.
Rationale: Re-reading, restating uncertainty, and refining an untested model do not
reduce environmental uncertainty.

Rule: Submit a reversible probe by the end of your second consecutive response
without an action, unless you identify a concrete irreversible risk, missing input,
or computation whose result will change that probe. A single action does not require
a complete helper, experiment record, objective theory, or multi-action prediction.
Rationale: The default cost of one probe is lower than repeated model calls. This is
a deadline implication, not permission for random action.

Rule: After observing a successor, repair only the contradicted, decision-relevant
claim, then choose the next discriminating action. Do not turn every correction into
an exhaustive world-model rewrite.
Rationale: The objective is timely orientation and level progress, not a perfect
description of the game.
'''

CRITIC_TEMPO_CONTRACT = r'''

Rule: First ask whether the Actor is delaying a cheap, legal, reversible action that
would create decision-relevant evidence. Criticize analysis that cannot produce new
evidence as readily as premature commitment.
Rationale: Your primary tempo duty is to expose when contact with the environment
will reduce uncertainty faster than another model call.

Rule: Use ACT_NOW when a specific reversible probe separates live alternatives more
cheaply than further thought. Use THINK_FIRST only when you can name a concrete
irreversible risk, missing input, or bounded computation whose result changes the
probe. Otherwise use CONTINUE.
Rationale: Uncertainty alone does not justify delay. ACT_NOW must identify what the
probe distinguishes; THINK_FIRST carries the burden of proof.

Keep the review epistemic. Do not verify game facts, debug code, solve the level, or
demand a complete theory. Return the smallest intervention that changes the timing
or warrant of the Actor's next commitment.
'''

FIELDS = {
    'assessment', 'tempo_verdict', 'actor_quote', 'inference_gap',
    'decision_consequence', 'minimal_revision',
}


def validate_tempo_critique(value):
    if not isinstance(value, dict) or set(value) != FIELDS:
        raise ValueError('six tempo-critique fields required')
    if not all(isinstance(v, str) for v in value.values()):
        raise ValueError('tempo-critique fields must be strings')
    if value['assessment'] not in ('concern', 'no_concern', 'insufficient_context'):
        raise ValueError('invalid epistemic assessment')
    if value['tempo_verdict'] not in ('ACT_NOW', 'THINK_FIRST', 'CONTINUE'):
        raise ValueError('invalid tempo verdict')
    limits = {
        'actor_quote': 25,
        'inference_gap': 32,
        'decision_consequence': 28,
        'minimal_revision': 32,
    }
    if any(len(value[field].split()) > limit for field, limit in limits.items()):
        raise ValueError('critic field word budget exceeded')
    if value['assessment'] == 'no_concern':
        if value['tempo_verdict'] != 'CONTINUE' or any(
            value[field].strip() for field in limits
        ):
            raise ValueError('no_concern must be empty CONTINUE')
    elif value['assessment'] == 'concern':
        if any(not value[field].strip() for field in limits):
            raise ValueError('concern requires specific nonempty fields')
    elif value['tempo_verdict'] == 'ACT_NOW':
        raise ValueError('insufficient_context cannot require ACT_NOW')
    return value

