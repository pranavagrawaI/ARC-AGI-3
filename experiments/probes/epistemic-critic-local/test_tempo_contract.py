import pytest

from tempo_contract import ACTOR_TEMPO_CONTRACT, validate_tempo_critique
from live_system_pivot_v4 import CRITIC_PROMPT


def critique(**changes):
    value = {
        'assessment': 'concern',
        'tempo_verdict': 'ACT_NOW',
        'actor_quote': 'I will inspect the grid again.',
        'inference_gap': 'Another inspection cannot create evidence about which action moves the object.',
        'decision_consequence': 'The delay consumes the deadline without distinguishing the live movement hypotheses.',
        'minimal_revision': 'Submit one reversible UP probe and compare its successor with the alternatives.',
    }
    value.update(changes)
    return value


def test_actor_contract_makes_action_the_default_after_two_responses():
    assert 'end of your second consecutive response' in ACTOR_TEMPO_CONTRACT
    assert 'A single action does not require' in ACTOR_TEMPO_CONTRACT


def test_critic_prompt_has_one_unambiguous_output_contract():
    assert 'exactly these six string fields' in CRITIC_PROMPT
    assert 'exactly these five fields' not in CRITIC_PROMPT
    assert 'ACT_NOW|THINK_FIRST|CONTINUE' in CRITIC_PROMPT


def test_valid_act_now_concern():
    assert validate_tempo_critique(critique())['tempo_verdict'] == 'ACT_NOW'


def test_no_concern_is_empty_continue():
    value = critique(assessment='no_concern', tempo_verdict='CONTINUE',
                     actor_quote='', inference_gap='', decision_consequence='',
                     minimal_revision='')
    assert validate_tempo_critique(value) == value


@pytest.mark.parametrize('changes', [
    {'tempo_verdict': 'WAIT'},
    {'assessment': 'no_concern', 'tempo_verdict': 'ACT_NOW', 'actor_quote': '',
     'inference_gap': '', 'decision_consequence': '', 'minimal_revision': ''},
    {'assessment': 'insufficient_context', 'tempo_verdict': 'ACT_NOW',
     'actor_quote': '', 'inference_gap': '', 'minimal_revision': ''},
])
def test_rejects_contradictory_status(changes):
    with pytest.raises(ValueError):
        validate_tempo_critique(critique(**changes))
