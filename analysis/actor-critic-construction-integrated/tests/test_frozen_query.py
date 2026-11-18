from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from frozen_client import FrozenModelClient  # noqa: E402  # identity path
from host import FixtureTransport, run_episode  # noqa: E402
from tests.test_host import CANDIDATE, WORK, call, case_from, response  # noqa: E402


def test_evaluator_is_not_ls20_observer() -> None:
    text = (HERE / "evaluate_visible.py").read_text()
    assert "observed(" not in text
    assert "bar_left" not in text
    assert "grid[61]" not in text
    assert "color 12" not in text.lower()


def test_actor_query_uses_sealed_rpc_after_freeze(tmp_path: Path) -> None:
    out = tmp_path / "out"
    record = run_episode(
        case_from(tmp_path),
        FixtureTransport([response([call("submit_candidate", {"work_item": WORK, "candidate_source": CANDIDATE})])]),
        out,
    )
    assert record["frozen"] is True
    client = FrozenModelClient(out)
    predicted = client.query_frozen_predict("RIGHT", {"grid": [[12, 4, 4]]}, step=1)
    assert "predicted" in predicted
    assimilated = client.assimilate_frozen(
        {"grid": [[4, 12, 4]]}, executed_action="RIGHT", executed_step=2
    )
    assert "unscored" in assimilated
    assert FrozenModelClient(tmp_path / "missing").query_frozen_predict("UP", step=0) == "NO_FROZEN_MODEL"


def _valid_freeze(tmp_path: Path) -> FrozenModelClient:
    source = "pass"
    (tmp_path / "FROZEN.json").write_text(json.dumps({
        "schema": "actor-critic-construction-frozen-v1",
        "candidate_source": source,
        "source_digest": hashlib.sha256(source.encode()).hexdigest(),
    }))
    return FrozenModelClient(tmp_path)


def test_query_reset_does_not_drop_pending(tmp_path: Path, monkeypatch) -> None:
    client = _valid_freeze(tmp_path)
    monkeypatch.setattr(client, "_rpc", lambda source: None)
    client._pending = {"grid": [[1]]}
    client._pending_key = ("UP", 1)
    assert client.query_frozen_predict("RESET", {"grid": [[1]]}, step=1) == "PREDICT_WITHOUT_ASSIMILATE"
    assert client._pending_key == ("UP", 1)


def test_executed_reset_drops_pending(tmp_path: Path) -> None:
    client = _valid_freeze(tmp_path)
    client._pending = {"grid": [[1]]}
    client._pending_key = ("UP", 1)
    client.note_executed({"action": "RESET", "after": {"levels_completed": 0, "grid": [[1]]}})
    assert client._pending is None
    assert client._pending_key is None


def test_assimilate_binds_executed_successor(tmp_path: Path, monkeypatch) -> None:
    client = _valid_freeze(tmp_path)
    calls = []

    class RPC:
        def assimilate(self, pending, observation, verdict):
            calls.append((pending, observation, verdict))
            return {"grid": observation["grid"]}

    monkeypatch.setattr(client, "_rpc", lambda source: RPC())
    client._pending = {"grid": [[12, 4, 4]]}
    client._pending_key = ("RIGHT", 4)
    assert client.assimilate_frozen({"grid": [[4, 12, 4]]}, executed_action="RIGHT", executed_step=4) == "REJECT_MISMATCH"
    assert client.assimilate_frozen({"grid": [[4, 12, 4]]}, executed_action="LEFT", executed_step=5) == "REJECT_MISMATCH"
    assert "unscored" in client.assimilate_frozen({"grid": [[4, 12, 4]]}, executed_action="RIGHT", executed_step=5)
    assert calls and calls[0][2] == {"status": "unscored"}
    assert client.assimilate_frozen({"grid": [[4, 12, 4]]}, executed_action="RIGHT", executed_step=6) == "AMBIGUOUS_PENDING"
