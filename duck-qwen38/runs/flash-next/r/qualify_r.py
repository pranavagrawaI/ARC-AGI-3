"""Offline qualification of arm R against KF: parser replay over out-kf2 plus ACTION7 dispatch.

Run with the duck-critic-adapter venv (it has arcengine):
  ~/projects/arc-agi-3-reference/duck-critic-adapter/ARC3-Inference/.venv/bin/python -B r/qualify_r.py
"""
import ast
import glob
import importlib.util
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FLASH = HERE.parent
INF = "source-dataset/src/ARC3-Inference/inference"
TRANSCRIPTS = FLASH / "out-kf2/artifacts"


def load(tree_root: Path, tag: str) -> dict:
    spec = importlib.util.spec_from_file_location(f"{tag}_action_names", tree_root / INF / "agent/action_names.py")
    names = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(names)
    import arcengine

    ns = {"re": re, "json": json, "Any": object, "arcengine": arcengine, "to_engine_action": names.to_engine_action}

    def take(path: Path, wanted: set) -> None:
        tree = ast.parse(path.read_text())
        fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in wanted]
        assert {f.name for f in fns} == wanted, (path, wanted - {f.name for f in fns})
        mod = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *fns], type_ignores=[])
        exec(compile(ast.fix_missing_locations(mod), str(path), "exec"), ns)

    take(tree_root / INF / "agent/tool_agent.py", {"_normalize_summary_text", "_extract_labeled_blocks", "_extract_scientist_note"})
    take(tree_root / INF / "framework/solver.py", {"_normalize_actions"})
    ns["names"] = names
    return ns


def assistant_blocks():
    for path in sorted(glob.glob(str(TRANSCRIPTS / "*events.jsonl"))):
        for line in open(path):
            event = json.loads(line)
            if event["type"] != "analysis":
                continue
            for m in re.finditer(
                r"^\[ASSISTANT\]\n(.*?)(?=^\[(?:TOOL CALL: python|MODEL RESPONSE META|THINKING|ANALYZER STATUS)\]|\Z)",
                event["transcript"], re.M | re.S,
            ):
                yield Path(path).name[:4], m[1]


def main() -> int:
    kf, r = load(FLASH / "kf", "kf"), load(HERE, "r")
    failures = []

    # 1. Replay: every authored block, both parsers.
    fields = ["World model", "Cross-level notes", "Plan", "Goal model", "Action model", "Recent findings", "Open questions"]
    label_re = {f: re.compile(rf"^[-*#\s]*\**\s*{re.escape(f)}[^\n:]{{0,40}}:", re.M | re.I) for f in fields}
    key = {"World model": "world_model", "Cross-level notes": "cross_level_notes", "Plan": "current_plan", "Goal model": "goal_model",
           "Action model": "action_model", "Recent findings": "recent_findings", "Open questions": "open_questions"}
    counts = {f: {"authored": 0, "kf_filled": 0, "r_filled": 0} for f in fields}
    regressions = 0
    for _, block in assistant_blocks():
        a, b = kf["_extract_scientist_note"](block), r["_extract_scientist_note"](block)
        for f in fields:
            if label_re[f].search(block):
                counts[f]["authored"] += 1
                counts[f]["kf_filled"] += bool(a.get(key[f]))
                counts[f]["r_filled"] += bool(b.get(key[f]))
        # R may only add fields; any field KF filled must still be filled.
        regressions += sum(1 for k, v in a.items() if v and not b.get(k))
    if regressions:
        failures.append(f"{regressions} fields filled by KF but empty under R")
    wm = counts["World model"]
    if wm["r_filled"] < 0.95 * wm["authored"]:
        failures.append(f"world model fill {wm['r_filled']}/{wm['authored']} below 95%")

    # 2. Unit cases: what must split, what must not.
    cases = {
        "World model (revised): x": ("world_model", "x"),
        "**World model:** x": ("world_model", "x"),
        "## Cross-level notes (VERIFIED): y": ("cross_level_notes", "y"),
        "World model L3 (revised): z": ("world_model", "z"),
        "World model: a\nHypothesis 2: stays in world model": ("world_model", "a Hypothesis 2: stays in world model"),
        "Plan: p\nPlanning: stays in plan": ("current_plan", "p Planning: stays in plan"),
    }
    for text, (field, want) in cases.items():
        got = r["_extract_scientist_note"](text).get(field)
        if got != want:
            failures.append(f"case {text!r}: {field}={got!r}, want {want!r}")

    # 3. ACTION7 through the real normalizer and arcengine enum; KF must still reject it.
    import arcengine

    r_actions, r_err = r["_normalize_actions"](None, {"actions": [{"action": "ACTION7"}, {"action": "UP"}]})
    kf_actions, kf_err = kf["_normalize_actions"](None, {"actions": [{"action": "ACTION7"}]})
    ok7 = r_err is None and [x.id for x in r_actions] == [arcengine.GameAction.ACTION7, arcengine.GameAction.ACTION1]
    if not ok7:
        failures.append(f"R ACTION7 dispatch: {r_actions} {r_err}")
    if kf_err is None:
        failures.append("KF control unexpectedly accepts ACTION7")
    bogus, bogus_err = r["_normalize_actions"](None, {"action": "ACTION8"})
    if bogus_err is None:
        failures.append("R accepts ACTION8")
    if r["names"].to_model_action("ACTION7") != "ACTION7":
        failures.append("model-facing ACTION7 label changed")

    report = {"parser": counts, "parser_regressions": regressions,
              "action7": {"r": [str(x.id) for x in r_actions or []], "kf_error": kf_err, "action8_error": bogus_err},
              "failures": failures, "pass": not failures}
    (HERE / "qualify_r.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
