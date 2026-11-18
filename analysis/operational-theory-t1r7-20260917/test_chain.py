"""Predecessor-chain provenance regressions.

The fresh T1R7 chain must resolve fully to real bytes; the malformed frozen T1R6
chain (C6-F1) must fail with the exact defects the reviewer found. This is the
regression that keeps an incomplete provenance record from freezing again.
"""

from bootstrap import HERE, ROOT
from chain import validate_chain

FROZEN_T1R6 = ROOT / "analysis/operational-theory-t1r6-20260917/PREDECESSOR.json"


def test_fresh_chain_fully_resolves():
    ok, errors, sequence = validate_chain(HERE / "PREDECESSOR.json", ROOT)
    assert ok, errors
    assert sequence == ["t1r6", "t1r5", "t1r4", "t1r3", "t1r2", "t1r1", "t1"]


def test_frozen_t1r6_chain_is_malformed():
    ok, errors, sequence = validate_chain(FROZEN_T1R6, ROOT)
    assert not ok
    joined = " ".join(errors)
    assert "missing file T2R3-REVIEW.json" in joined
    assert "hash mismatch PREDECESSOR.json" in joined
    assert "sequence skip t1r5 -> t1r2" in joined
    assert sequence == ["t1r5", "t1r2", "t1r1", "t1"]
