from pathlib import Path


def test_v3_hook_uses_cacheable_prefix():
    source=(Path(__file__).resolve().parent.parent/'live_hook_v2.py').read_text()
    assert "Context(seed, prefix=True)" in source
    assert "Context(seed, prefix=False)" not in source


def test_marker_gate_precedes_spend_reservation():
    source=(Path(__file__).resolve().parent.parent/'live_system_pivot_v3.py').read_text()
    assert source.index("Actor evidence absent before reservation") < source.index("budget.reserve(reserve)")
