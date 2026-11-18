"""Score a Q2 run against PRECOMMIT.md. Written before the run finished.

Scores the precommitted metrics exactly as they were defined. Where seeing partial data
suggested a sharper metric, that metric is reported separately and labelled post-hoc, so
it cannot quietly become the gate.

    python3 score_q2.py <run-dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# The skill's own vocabulary. A cell that calls one of these is working on the game;
# a cell that only calls doc/print/pprint is reading the API or echoing.
GRID_FUNCTIONS = {
    "grid_array", "trajectory", "render_grid", "write_helper", "load_helpers",
    "np", "numpy", "array", "argwhere", "unique", "nonzero", "where",
}
# Paths an admissible contestant must never read (CONTEXT.md). --sandbox drop needed
# root, so this grep replaces the uid backstop rather than merely double-checking it.
FORBIDDEN = ("environment_files", "ls20.py", "/proc/", "arc_agi_3/environment")


def load_events(run: Path) -> list[dict]:
    path = next(run.glob("agent_logs/**/events.jsonl"), None)
    return [json.loads(l) for l in path.read_text().splitlines()] if path else []


def load_actions(run: Path) -> list[dict]:
    path = (run / "ipc/actions.jsonl").resolve()
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def emitted_cells(run: Path) -> list[str]:
    """Distinct CodeAct cell source the model emitted, from the OTel spans."""
    import glob
    out: list[str] = []
    for f in sorted((run / "traces").resolve().glob("*.jsonl")):
        for line in f.read_text().splitlines():
            for rs in json.loads(line).get("resourceSpans", []):
                for ss in rs.get("scopeSpans", []):
                    for sp in ss.get("spans", []):
                        for a in sp.get("attributes", []):
                            if a["key"].startswith("llm.output_messages") and (
                                "function.arguments" in a["key"]
                            ):
                                v = a.get("value", {}).get("stringValue", "")
                                try:
                                    code = json.loads(v).get("code") or v
                                except Exception:
                                    code = v
                                if code not in out:
                                    out.append(code)
    return out


def cell_sources(run: Path) -> list[str]:
    """Cell source text, recovered from the OTel spans (events record only a length)."""
    out: list[str] = []
    for trace in (run / "traces").resolve().glob("*.jsonl"):
        for line in trace.read_text().splitlines():
            blob = json.dumps(json.loads(line))
            if "def " in blob or "grid_array" in blob or "import " in blob:
                out.append(blob)
    return out


def score(run: Path) -> dict:
    events = load_events(run)
    actions = load_actions(run)
    helpers = sorted((run / "team_nemo/shared/helpers").resolve().glob("*.py"))
    knowledge = sorted(
        p for p in (run / "team_nemo/shared/knowledge").resolve().glob("**/*") if p.is_file()
    )

    steps = sorted({e["step"] for e in events if "step" in e})
    repl = [e for e in events if e["event"] == "repl_execute"]
    llm = [e for e in events if e["event"] == "llm_call"]

    # --- precommitted metrics, as defined -----------------------------------------
    steps_with_repl = {e["step"] for e in repl}
    compute_rate = len(steps_with_repl) / len(steps) if steps else 0.0

    helper_text = "\n".join(p.read_text() for p in helpers)
    has_encode_predict = "def encode" in helper_text and "def predict" in helper_text

    submissions_per_step: dict[int, int] = {}
    for a in actions:
        submissions_per_step[a.get("turn", a.get("step", -1))] = (
            submissions_per_step.get(a.get("turn", a.get("step", -1)), 0) + 1
        )
    # PRECOMMIT says "turns ending in exactly one submit_actions / turns". Dividing by
    # len(actions) instead scores 1.0 whenever every submission is well formed, even if
    # the agent submitted on one turn out of ten — it measures the submissions, not the
    # turns. Divide by env steps, which is what a turn is.
    env_steps = sum(1 for e in events if e["event"] == "env_step")
    contract_ok = sum(1 for n in submissions_per_step.values() if n == 1)
    turn_contract = contract_ok / env_steps if env_steps else 0.0

    rationales = [str(a.get("rationale", "")) for a in actions]
    retrodiction = (
        sum(1 for r in rationales if r.strip().lower().startswith("predict"))
        / len(rationales)
        if rationales
        else 0.0
    )
    batch_sizes = [len(a.get("actions", []) or []) for a in actions]

    # --- containment: replaces the uid drop this run could not use ----------------
    leaks = [f for blob in cell_sources(run) for f in FORBIDDEN if f in blob]

    # --- post-hoc, NOT the gate ---------------------------------------------------
    # `functions_used` in events.jsonl does NOT list calls the cell makes on `self`:
    # cells that plainly call self.grid_array / np.unique are recorded as
    # ['print', 'pprint', 'doc']. Reading grid work off that field says the opposite of
    # the truth, so this metric reads the emitted cell source from the OTel spans.
    emitted = emitted_cells(run)
    grid_cells = [c for c in emitted if any(f in c for f in GRID_FUNCTIONS)]
    steps_with_grid_work = set()  # emitted cells are not step-tagged in the spans

    gate = {
        "compute_rate>=0.50": compute_rate >= 0.50,
        "encode_and_predict_helper": has_encode_predict,
        "turn_contract>=0.90": turn_contract >= 0.90,
    }
    inconclusive = len(steps) < 15
    verdict = (
        "INCONCLUSIVE (fewer than 15 turns)"
        if inconclusive
        else ("PASS" if all(gate.values()) else "FAIL")
    )
    if leaks:
        verdict = "VOID — game source referenced in an executed cell"

    return {
        "run": run.name,
        "verdict": verdict,
        "gate": gate,
        "precommitted": {
            "turns": len(steps),
            "env_steps": env_steps,
            "empty_turn_answers": sum(
                1 for e in events
                if e["event"] == "step_complete" and not (e.get("answer") or "").strip()
            ),
            "compute_rate": round(compute_rate, 3),
            "helpers": [p.name for p in helpers],
            "encode_and_predict": has_encode_predict,
            "turn_contract": round(turn_contract, 3),
            "retrodiction_rate": round(retrodiction, 3),
            "knowledge_files": [p.name for p in knowledge],
            "knowledge_bytes": sum(p.stat().st_size for p in knowledge),
            "submissions": len(actions),
            "batch_sizes": batch_sizes,
        },
        "containment": {"forbidden_hits": sorted(set(leaks)), "clean": not leaks},
        "post_hoc_not_the_gate": {
            "cells_executed": len(repl),
            "distinct_cells_emitted": len(emitted),
            "emitted_cells_touching_the_grid": len(grid_cells),
            "grid_cell_share_of_emitted": round(
                len(grid_cells) / len(emitted), 3
            ) if emitted else 0.0,
            "note": "measured from emitted cell source; events.jsonl functions_used "
                    "omits self.* calls and reads as doc/print for grid-computing cells",
        },
        "cost": {
            "llm_calls": len(llm),
            "output_tokens": sum(e.get("output_tokens") or 0 for e in llm),
            "reasoning_tokens": sum(e.get("reasoning_tokens") or 0 for e in llm),
            "input_tokens": sum(e.get("input_tokens") or 0 for e in llm),
        },
    }


if __name__ == "__main__":
    print(json.dumps(score(Path(sys.argv[1])), indent=1))
