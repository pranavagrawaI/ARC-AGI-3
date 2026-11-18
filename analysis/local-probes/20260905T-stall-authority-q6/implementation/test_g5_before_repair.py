#!/usr/bin/env python3
"""Executable assertions for the q6/G5 stall interlock. No model, no environment.

The gate is mechanical by design, so it can be checked mechanically. These cover
the three leakage paths that would silently void the experiment:

  * a stall never fires, so the run has no treatment;
  * a stall fires but a submission slips through unresolved;
  * `continue` is accepted without a falsifier, which is the ceremony the card
    exists to catch.

Run:  uv run --frozen python analysis/local-probes/20260905T-stall-authority-q6/test_g5_offline.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

AFTER = Path(__file__).resolve().parent / "patched-source" / "after"


def load_chassis():
    sys.path.insert(0, str(AFTER))
    spec = importlib.util.spec_from_file_location("q6_solver_agent", AFTER / "solver_agent.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # `hidden` resolves its calling module from sys.modules, so register first.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stub:
    """Only the attributes the gate touches. Anything more would test the framework."""

    def __init__(self, tmp: Path, mod) -> None:
        self._mod = mod
        tmp.mkdir(parents=True, exist_ok=True)
        self.ipc_dir = tmp
        self._actions_path = tmp / "actions.jsonl"
        self._authority_path = tmp / "authority.jsonl"
        self._actions_path.touch()
        self._stall_id = 0
        self._resolved_stall_id = 0
        self._stall_since_action = None
        self._stall_repeated_policy = ""
        self._stall_evaluated_turn = None
        self.rejections: list[tuple[str, str]] = []

    _authority_event = None  # bound below
    _recent_batches = None
    _evaluate_stall = None
    resolve_stall = None

    def _log_rejection(self, reason, actions, detail):
        self.rejections.append((reason, detail))

    def events(self) -> list[dict]:
        if not self._authority_path.exists():
            return []
        return [json.loads(x) for x in self._authority_path.read_text().splitlines() if x.strip()]


def bind(stub: Stub, mod) -> Stub:
    cls = mod.ArcSolverBase
    for name in ("_authority_event", "_recent_batches", "_evaluate_stall",
                 "_stall_status_line", "resolve_stall"):
        setattr(type(stub), name, getattr(cls, name))
    return stub


def write_batches(stub: Stub, rationales: list[str], actions: list[str], levels: int = 0) -> None:
    with stub._actions_path.open("w") as handle:
        for i, text in enumerate(rationales):
            handle.write(json.dumps({
                "turn": i, "actions": [actions[i % len(actions)]],
                "rationale": text, "levels_completed": levels,
            }) + "\n")


def state(step: int, turn: int, levels: int = 0, available=None) -> dict:
    return {
        "turn": turn, "step": step, "level": levels, "state": "NOT_FINISHED",
        "levels_completed": levels,
        "available_actions": available or ["UP", "DOWN", "LEFT", "RIGHT", "RESET"],
        "action_results": [],
    }


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"  -- {detail}" if detail and not condition else ""))
    if not condition:
        raise SystemExit(1)


def main() -> int:
    mod = load_chassis()
    tmp = Path(tempfile.mkdtemp())

    print("G5 stall interlock")

    # 1. Repeating policy with no progress and an untried action -> stall opens.
    s = bind(Stub(tmp / "a", mod), mod)
    write_batches(s, ["predict: navigate toward the spiral"] * 6, ["UP", "DOWN"])
    s._evaluate_stall(state(step=40, turn=6))
    check("stall opens on repeated policy without progress", s._stall_id == 1)
    detected = [e for e in s.events() if e["event"] == "STALL_DETECTED"]
    check("STALL_DETECTED carries since_action and untried actions",
          bool(detected) and detected[0]["since_action"] == 40 and detected[0]["untried_actions"])

    # 2. Below the action floor it must stay quiet.
    s2 = bind(Stub(tmp / "b", mod), mod)
    write_batches(s2, ["predict: navigate toward the spiral"] * 6, ["UP", "DOWN"])
    s2._evaluate_stall(state(step=10, turn=6))
    check("no stall below the action floor", s2._stall_id == 0)

    # 3. Progress inside the window must clear it.
    s3 = bind(Stub(tmp / "c", mod), mod)
    with s3._actions_path.open("w") as handle:
        for i in range(6):
            handle.write(json.dumps({"turn": i, "actions": ["UP"],
                                     "rationale": "predict: navigate toward the spiral",
                                     "levels_completed": 0 if i < 3 else 1}) + "\n")
    s3._evaluate_stall(state(step=40, turn=6, levels=1))
    check("progress inside the window suppresses the stall", s3._stall_id == 0)

    # 4. resolve_stall shape rules.
    check("continue without abandon_if is refused",
          s.resolve_stall("continue", "H2", "H2").startswith("REJECTED"))
    check("switch to the same basis is refused",
          s.resolve_stall("switch", "H2", "H2").startswith("REJECTED"))
    check("unknown disposition is refused",
          s.resolve_stall("ponder", "H2", "H7").startswith("REJECTED"))
    check("stall still open after refused resolutions", s._stall_id != s._resolved_stall_id)
    ok = s.resolve_stall("switch", "H2", "H7")
    check("valid switch resolves the interlock", s._stall_id == s._resolved_stall_id, ok)
    resolved = [e for e in s.events() if e["event"] == "STALL_RESOLVED"]
    check("STALL_RESOLVED records both bases",
          resolved and resolved[0]["previous_basis"] == "H2" and resolved[0]["basis"] == "H7")
    check("resolving twice is refused", s.resolve_stall("switch", "H7", "H8").startswith("REJECTED"))

    # 5. continue WITH a falsifier is accepted, and records it.
    s5 = bind(Stub(tmp / "d", mod), mod)
    write_batches(s5, ["predict: navigate toward the spiral"] * 6, ["UP", "DOWN"])
    s5._evaluate_stall(state(step=40, turn=6))
    s5.resolve_stall("continue", "H2", "H2", abandon_if="the spiral does not move after UP")
    check("continue with a falsifier is accepted", s5._stall_id == s5._resolved_stall_id)
    kept = [e for e in s5.events() if e["event"] == "STALL_RESOLVED"][0]
    check("abandon_if is recorded verbatim", kept["abandon_if"].startswith("the spiral"))

    # 6. The interlock condition itself, as the submission gate reads it.
    src = (AFTER / "solver_agent.py").read_text()
    check("submit_actions gates on the exact id comparison",
          "if self._stall_id != self._resolved_stall_id:" in src
          and '_log_rejection("stall_unresolved"' in src)
    check("G5 precedes the batching gate",
          src.index('_log_rejection("stall_unresolved"')
          < src.index("# G3/G4 — batch length is earned"))

    print("\nall G5 assertions hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
