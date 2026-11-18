import pytest
from live_endurance import Budget, AdmissionStop, review_due


def test_completed_calls_release_unused_reservation():
    b=Budget(.5)
    for _ in range(20):
        b.reserve(.1);b.settle(.1,.01)
    assert b.spent==pytest.approx(.2)
    assert b.pending==pytest.approx(0)
    with pytest.raises(AdmissionStop):b.reserve(.31)


def test_unknown_cost_keeps_exposure_reserved():
    b=Budget(.5);b.reserve(.4)
    with pytest.raises(ValueError):b.settle(.4,None)
    with pytest.raises(AdmissionStop):b.reserve(.2)
    assert b.pending==.4


def test_repeated_review_requires_new_calls_or_actions_and_stops_at_cap():
    assert not review_due(3,0,0,0,0)
    assert review_due(4,0,0,0,0)
    assert not review_due(5,0,4,0,1)
    assert review_due(5,4,4,0,1)
    assert not review_due(50,40,0,0,12)
