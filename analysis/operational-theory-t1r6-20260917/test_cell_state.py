"""Real worker/Actor/CodeAct regressions, loopback responses and raw evidence only."""

import hashlib
import json
from pathlib import Path

import pytest

from bootstrap import ROOT
from identity import sha
from test_adversarial import run, outputs

HERE = Path(__file__).parent
SEQUENCE = json.loads((HERE / "t3-treatment-sequence.json").read_text())
END = "self.submit_actions(['UP'], rationale='cell state fixture')"


def user_output(request):
    return "\n".join(
        m.get("content") or ""
        for m in request["messages"]
        if m["role"] == "user" and "PythonOutput(" in (m.get("content") or "")
    )


def test_exact_t3_treatment_replay(tmp_path):
    for row in SEQUENCE:
        source = ROOT / row["response"]
        assert sha(source) == row["sha256"]
        tool = json.loads(source.read_bytes())["choices"][0]["message"]["tool_calls"][0]
        assert tool == row["tool_call"]
        assert json.loads(tool["function"]["arguments"])["code"] == row["code"]
    result, requests = run(
        tmp_path,
        "treatment",
        [s["code"] for s in SEQUENCE],
        [s["tool_call"]["id"] for s in SEQUENCE],
    )
    assert len(requests) == 6
    out = user_output(requests[5])
    assert "NameError" not in out and "CELL_STATE_REPAIR" not in out
    # Verify cell 5 actually read the schema, not merely that its error disappeared.
    assert "revision_if_wrong" in out and "Cell In[5]" not in out
    events = [
        json.loads(s)
        for s in (tmp_path / "treatment/host-events.jsonl").read_text().splitlines()
    ]
    reads = [
        e for e in events if e["op"] == "theory" and e["request"]["operation"] == "read"
    ]
    assert len(reads) == 3  # cells 4, 5, 6 (old runtime: only 4 and 6)
    assert not result["metrics"][
        "construction_completion"
    ]  # unchanged sequence never proposes
    (tmp_path / "EXACT-REPLAY.json").write_text(
        json.dumps(
            dict(
                provider_calls=0,
                exact_cells=6,
                successful_schema_reads=3,
                cell5="success",
                code_sha256=[
                    hashlib.sha256(s["code"].encode()).hexdigest() for s in SEQUENCE
                ],
            ),
            indent=2,
        )
    )


@pytest.mark.parametrize("arm", ["control", "treatment"])
def test_user_binding_persistence(tmp_path, arm):
    codes = [
        "import json as _json\nimport numpy as _np\n_arr = _np.array([2,3,5])\nordinary = 7\n_gvar = 11\n__user = 13\ndef _f(value):\n    return value + 1\n_alias = _f\n_method = _arr.sum\n",
        "_json = _json\n_alias = _alias\n_arr = _arr * 2\nordinary += 1\nprint('PERSIST', _json.dumps([_arr.tolist(), _alias(4), int(_method()), ordinary, _gvar, __user]))\n",
        "print('NEXT', _json.dumps([_arr.tolist(), _f(8), ordinary]))\n"
        + (
            END
            if arm == "control"
            else "print(self.operational_cycle('read')['phase'])"
        ),
    ]
    if arm == "control":
        codes.append(END)
    else:
        codes.extend(['print("padding")'] * 3)
    result, requests = run(tmp_path, arm, codes)
    out = outputs(requests)
    assert "PERSIST [[4, 6, 10], 5, 10, 8, 11, 13]" in out
    assert "NEXT [[4, 6, 10], 9, 8]" in out
    assert "CELL_STATE_REPAIR" not in out and "NameError" not in out
    assert result["metrics"]["construction_completion"] == (arm == "control")


@pytest.mark.parametrize(
    "attack",
    [
        "_CELL_BANNED_NAMES = set()",
        "__repl_captured_locals__ = {'leak': self}",
        "import json as __repl_wrapper__",
        "self = 'replaced'",
        "import json as notification",
        "_leak = __repl_captured_locals__; print(_leak)",
        "_leak = self._model_feedback; print(_leak)",
        "self._model_feedback = None",
        "_leak = globals(); print(_leak)",
        "import nooa.runtime.actor as _host; print(_host)",
        "_leak = getattr(self, '_' + 'model_feedback'); print(_leak)",
        "def _leak(x=_CELL_BANNED_NAMES):\n    return x\nprint(_leak())",
    ],
)
def test_private_binding_denied_without_execution_or_persistence(tmp_path, attack):
    codes = [
        "open('/work/ipc/ALLOWED_CONTROL','w').write('ok')\n_retained = 'kept'",
        "open('/work/ipc/ATTACK_EXECUTED','w').write('bad')\n" + attack,
        "print('INTACT', _retained, self.latest_state()['step'])\n" + END,
        END,
    ]
    result, requests = run(
        tmp_path,
        "control",
        codes,
        ["first", "local_prefill_deadbeef", "third", "fourth"],
    )
    assert result["metrics"]["construction_completion"]
    assert len(requests) == 4
    out = outputs(requests)
    assert "CELL_STATE_REPAIR" in out and "cell not executed" in out
    assert "INTACT kept" in out
    assert (tmp_path / "control/actor/ipc/ALLOWED_CONTROL").read_text() == "ok"
    assert not (tmp_path / "control/actor/ipc/ATTACK_EXECUTED").exists()
    assert "execution_status=<ResultStatus.ERROR" not in out


def test_missing_binding_stops_before_side_effects_and_allows_local_repair(tmp_path):
    result, requests = run(
        tmp_path,
        "control",
        [
            "open('/work/ipc/SHOULD_NOT_EXECUTE','w').write('bad')\nprint(_absent)",
            "_absent = 19\nprint('FIXED',_absent)\n" + END,
            "self.submit_actions(['UP'], rationale='finish')\n" + END,
            "self.submit_actions(['UP'], rationale='finish2')\n" + END,
            "self.submit_actions(['UP'], rationale='finish3')\n" + END,
            "self.submit_actions(['UP'], rationale='finish4')\n" + END,
            "self.submit_actions(['UP'], rationale='finish5')\n" + END,
        ],
    )
    assert result["metrics"]["construction_completion"]
    assert len(requests) == 3
    out = outputs(requests)
    assert (
        "CELL_STATE_REPAIR: binding unavailable: _absent" in out and "FIXED 19" in out
    )
    assert not (tmp_path / "control/actor/ipc/SHOULD_NOT_EXECUTE").exists()


def test_deleted_binding_is_not_resurrected(tmp_path):
    result, requests = run(
        tmp_path,
        "control",
        [
            "_gone = 23\nkept = 29",
            "del _gone",
            "print(_gone)",
            "print('KEPT',kept)\n" + END,
            "self.submit_actions(['UP'], rationale='finish')\n" + END,
            "self.submit_actions(['UP'], rationale='finish2')\n" + END,
            "self.submit_actions(['UP'], rationale='finish3')\n" + END,
            "self.submit_actions(['UP'], rationale='finish4')\n" + END,
            "self.submit_actions(['UP'], rationale='finish5')\n" + END,
        ],
    )
    assert result["metrics"]["construction_completion"]
    assert "binding unavailable: _gone" in outputs(requests)
    assert "KEPT 29" in outputs(requests)


def test_binding_state_isolated_between_arms_and_fresh_sessions(tmp_path):
    (tmp_path / "first").mkdir()
    run(tmp_path / "first", "control", ["_only_here = 'secret fixture'\n" + END, END])
    for index, arm in enumerate(["treatment", "control"]):
        codes = ["print(_only_here)", "_only_here = 'new'\nprint('LOCAL',_only_here)"]
        codes += [END, END] if arm == "control" else ['print("padding")'] * 4
        (tmp_path / str(index)).mkdir()
        result, requests = run(tmp_path / str(index), arm, codes)
        out = outputs(requests)
        assert "binding unavailable: _only_here" in out and "LOCAL new" in out
        assert "secret fixture" not in json.dumps(requests)
        assert result["metrics"]["construction_completion"] == (arm == "control")


def test_runtime_missing_ordinary_name_has_bounded_feedback(tmp_path):
    result, requests = run(
        tmp_path,
        "control",
        [
            "kept = 31",
            "print(absent_public)",
            "print('RETAINED',kept)\n" + END,
            "self.submit_actions(['UP'], rationale='finish')\n" + END,
        ],
    )
    assert result["metrics"]["construction_completion"]
    assert len(requests) == 4
    out = outputs(requests)
    assert "CELL_STATE_REPAIR: binding unavailable:" in out
    assert "no retry performed" in out and "RETAINED 31" in out
    assert "execution_status=<ResultStatus.ERROR" not in out


def test_public_import_can_be_rebound_without_replacing_host_module(tmp_path):
    result, requests = run(
        tmp_path,
        "control",
        [
            "import json\n_json = json\njson = {'model': 41}",
            "print('PUBLIC_REBOUND',json['model'],_json.dumps([1,2]))\n" + END,
            "self.submit_actions(['UP'], rationale='finish')\n" + END,
        ],
    )
    assert result["metrics"]["construction_completion"]
    assert "PUBLIC_REBOUND 41 [1, 2]" in outputs(requests)
    assert "CELL_STATE_REPAIR" not in outputs(requests)


def test_t6_detached_function_globals_follow_rebinding(tmp_path):
    result, requests = run(
        tmp_path,
        "control",
        [
            "_value = 1\ndef _read():\n    return _value",
            "_value = 2\nprint('LIVE_FUNCTION', _read())\nself.submit_actions(['UP'], rationale='rebind')\n"
            + END,
        ]
        + ["self.submit_actions(['UP'], rationale='finish')\n" + END] * 5,
    )
    assert result["metrics"]["construction_completion"]
    # A retained function must observe the rebinding, not a stale snapshot: after
    # `_value = 2`, `_read()` returns 2. Asserting the value (not merely the label)
    # is what distinguishes the live namespace from the detached-globals failure.
    assert "LIVE_FUNCTION 2" in outputs(requests)
    assert "LIVE_FUNCTION 1" not in outputs(requests)
    assert "CELL_STATE_REPAIR" not in outputs(requests)


@pytest.mark.parametrize(
    "probe",
    [
        "import operator; _private = operator.attrgetter('_model_feedback')(self)",
        "import operator; _globals = operator.attrgetter('__globals__')(self.latest_state)",
        "import operator as op; _private = op.methodcaller('__getattribute__', '_model_feedback')(self)",
        "import builtins as b; _private = b.object.__getattribute__(self, '_model_feedback')",
        "_private = getattr(self, ''.join(['_', 'model_feedback']))",
        "_private = type(self).__mro__",
        "_private = self.latest_state.__class__.__mro__",
        "import pickle; _private = pickle.loads(pickle.dumps(self))",
        "from __future__ import annotations\n_private = self._model_feedback",
    ],
)
def test_t6_indirect_traversal_is_rejected_before_host_object_acceptance(
    tmp_path, probe
):
    result, requests = run(
        tmp_path,
        "control",
        [
            probe,
            "print('AFTER_TRAVERSAL', self.latest_state()['step'])\n" + END,
            END,
        ],
    )
    assert result["metrics"]["construction_completion"]
    out = outputs(requests)
    assert "CELL_STATE_REPAIR" in out
    assert "AFTER_TRAVERSAL 1" in out
    assert "PredictionFeedback" not in out and "__builtins__" not in out


def test_facade_public_tool_and_opaque_callable_work(tmp_path):
    result, requests = run(
        tmp_path,
        "control",
        [
            "state = self.latest_state()\nprint('FACADE_STEP', state['step'])",
            "print('FACADE_OBS', self.observations()['status'])\nself.submit_actions(['UP'], rationale='facade')\n"
            + END,
        ]
        + ["self.submit_actions(['UP'], rationale='finish')\n" + END] * 5,
    )
    assert result["metrics"]["construction_completion"]
    out = outputs(requests)
    assert "FACADE_STEP 1" in out and "FACADE_OBS" in out
    assert "CELL_STATE_REPAIR" not in out


SHUTDOWN_SEQUENCE = [
    "async def _background_reader():\n    open('/work/ipc/SHUTDOWN_STARTED','w').write('started')\n    try:\n        while True:\n            await asyncio.sleep(60)\n            yield 'waiting'\n    finally:\n        try:\n            open('/work/ipc/SHUTDOWN_READ','w').write(str(self.latest_state()['step']))\n        except Exception as _err:\n            open('/work/ipc/SHUTDOWN_ERROR','w').write(str(_err))\nself.queue_manager.spawn(_background_reader(),channel='system_messages',label='shutdown fixture')\nawait asyncio.sleep(0.01)\nprint(self.queue_manager.status())\nself.submit_actions(['UP'], rationale='shutdown fixture')\n",
    "self.submit_actions(['UP'], rationale='second')",
]


def test_background_broker_reader_joins_before_done(tmp_path):
    result, requests = run(tmp_path, "control", SHUTDOWN_SEQUENCE)
    assert len(requests) == 2 and result["metrics"]["construction_completion"]
    arm = tmp_path / "control"
    assert (arm / "actor/ipc/SHUTDOWN_STARTED").read_text() == "started"
    assert (arm / "actor/ipc/SHUTDOWN_READ").read_text() == "2"
    assert not (arm / "actor/ipc/SHUTDOWN_ERROR").exists()
    supervision = json.loads((arm / "SUPERVISION.json").read_text())
    assert supervision["reaped"] and supervision["returncode"] == 0
    events = [
        json.loads(s) for s in (arm / "host-events.jsonl").read_text().splitlines()
    ]
    assert events[-1]["op"] == "done"
    assert any(
        e["op"] == "evidence"
        for e in events[
            events.index(next(e for e in reversed(events) if e["op"] == "proposed"))
            + 1 : -1
        ]
    )
