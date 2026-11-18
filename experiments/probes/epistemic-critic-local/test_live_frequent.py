from live_frequent import review_due, Budget
import pytest

def test_reviews_recur_on_new_activity_and_stop_at_cap():
    assert not review_due(0,0,0,0,0)
    assert not review_due(1,0,0,0,0)
    assert review_due(2,0,0,0,0)
    assert not review_due(3,0,2,0,1)
    assert review_due(3,2,2,0,1)
    assert not review_due(50,40,0,0,20)

def test_reservations_preserve_spend_cap():
    b=Budget(.5)
    for _ in range(20):
        b.reserve(.1);b.settle(.1,.01)
    assert b.spent==pytest.approx(.2)
    assert b.pending==pytest.approx(0)
