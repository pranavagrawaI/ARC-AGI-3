import pytest
from paired_commitment import quote_binding, validate_critique, CRITIC_PROMPT


def test_quote_binding_resolves_repeated_occurrences_without_picking_one():
    assert quote_binding('claim','claim then claim')=={
        'status':'ambiguous','spans':[[0,5],[11,16]],'unit':'Unicode code points; end exclusive'}
    assert quote_binding('missing','claim')['status']=='absent'
    assert quote_binding('claim','a claim')['spans']==[[2,7]]


def test_execution_failure_is_not_an_epistemic_assessment():
    value=dict(assessment='timeout',actor_quote='',inference_gap='',decision_consequence='',minimal_revision='')
    with pytest.raises(ValueError):validate_critique(value)
    value.update(assessment='insufficient_context',decision_consequence='Meaning unclear.')
    assert validate_critique(value)==value


def test_generic_critic_does_not_disclose_selected_diagnosis():
    assert 'alignment' not in CRITIC_PROMPT
    assert 'knowing target' not in CRITIC_PROMPT
    assert 'known fault' not in CRITIC_PROMPT


def test_concern_requires_a_quotation_and_specific_correction():
    value=dict(assessment='concern',actor_quote='',inference_gap='Scope exceeds evidence.',
               decision_consequence='Abandons alternatives.',minimal_revision='Narrow scope.')
    with pytest.raises(ValueError):validate_critique(value)
