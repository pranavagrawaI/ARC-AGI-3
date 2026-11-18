"""Read saved transcripts; never execute model code or contact an endpoint.

Counts are evidence locators, not estimates of wasted tokens or causal score loss.
Repeated definitions are AST-identical within a game; repeated experiments may be
useful because the board can change. JSON tool payloads and plain payloads differ.
"""
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from transcript_stats import blocks

ROOTS = {
    "dfranzen": ROOT / "runs/out-control",
    "lordhansolo": ROOT / "lordhansolo/out",
    "sirikilohit": ROOT / "sirikilohit/out",
}


def code_from_call(body):
    match = re.search(r"<parameter=code>\s*\n?(.*?)</parameter>", body, re.S)
    return match.group(1).strip() if match else None


def error_from_result(body):
    try:
        payload = json.loads(body)
    except (ValueError, TypeError):
        payload = None
    if isinstance(payload, dict):
        error = payload.get("error", "")
        return str(error).replace("\\n", "\n") if error else ""
    if "Traceback" not in body:
        return ""
    return body[body.index("Traceback"):]


def analyze(root):
    totals = Counter()
    errors = Counter()
    examples = {}
    repeated_errors = []
    repeated_defs = []
    inputs = []
    for path in sorted((root / "transcripts").glob("*.txt")):
        data = path.read_bytes()
        inputs.append({"path": str(path), "sha256": hashlib.sha256(data).hexdigest()})
        local_errors = Counter()
        definitions = defaultdict(list)
        last_call = ""
        call_index = 0
        for kind, body in blocks(data.decode()):
            if kind == "TOOL CALL: python":
                totals["tool_calls"] += 1
                call_index += 1
                last_call = body
                code = code_from_call(body)
                if code is not None:
                    try:
                        tree = ast.parse(code)
                    except SyntaxError:
                        totals["unparsed_call_code"] += 1
                    else:
                        totals["parsed_call_code"] += 1
                        # Only top-level definitions, not code embedded as text.
                        for node in tree.body:
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                key = ast.dump(node, include_attributes=False)
                                definitions[key].append({"call": call_index, "name": node.name,
                                    "source": ast.get_source_segment(code, node)})
            elif kind == "TOOL RESULT: python":
                totals["tool_results"] += 1
                error = error_from_result(body)
                if not error:
                    continue
                matches = re.findall(r"^(?:\w+Error|\w+Exception):.*$", error, re.M)
                signature = matches[-1] if matches else error.splitlines()[-1]
                if signature.startswith(("TerminalStateActionError:", "StaleStateActionError:",
                        "KnownNoOpActionError:", "KnownDeathActionError:")):
                    category = "control_guard"
                elif ("failed save" in error or "checks failed" in error
                        or "executed no check" in error or "A save needs at least one test suite" in error):
                    category = "module_validation_rejection"
                else:
                    category = "other_tool_error"
                totals[category] += 1
                errors[signature] += 1
                local_errors[signature] += 1
                examples.setdefault(signature, {"path": str(path), "call_index": call_index,
                    "category": category, "call": last_call, "result": body})
        for signature, count in local_errors.items():
            if count > 1:
                repeated_errors.append({"path": str(path), "count": count, "error": signature})
        for occurrences in definitions.values():
            if len(occurrences) > 1:
                repeated_defs.append({"path": str(path), "count": len(occurrences),
                    "name": occurrences[0]["name"], "calls": [x["call"] for x in occurrences],
                    "source": occurrences[0]["source"]})
    totals["repeated_top_level_function_definitions"] = sum(x["count"] - 1 for x in repeated_defs)
    return {"counts": dict(totals), "inputs": inputs, "error_counts": dict(errors.most_common()),
        "error_examples": examples,
        "repeated_same_game_errors": sorted(repeated_errors, key=lambda x: -x["count"]),
        "repeated_definitions": sorted(repeated_defs, key=lambda x: -x["count"])}


def main():
    result = {"scope": "Dfranzen local public-25 control; lordhansolo and sirikilohit published short demos. Unequal exposure; no cross-donor error-rate ranking.",
        "limits": ["Counters are not a waste percentage. Guards and informative test failures can be useful.",
            "Duplicate definitions can be deliberate; no gameplay benefit is inferred from deleting them.",
            "Function scan excludes definitions inside saved-module source strings.",
            "Repeated visible states do not establish identical hidden game state."],
        "donors": {name: analyze(root) for name, root in ROOTS.items()}}
    target = HERE / "six-questions-evidence.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    for name, donor in result["donors"].items():
        print(name, json.dumps(donor["counts"]))
        print("repeated helpers:", [(d["path"].split('/')[-1][:4], d["name"], d["count"]) for d in donor["repeated_definitions"][:8]])
        print("repeated errors:", [(d["path"].split('/')[-1][:4], d["count"], d["error"]) for d in donor["repeated_same_game_errors"][:6]])
    print(target)


if __name__ == "__main__":
    main()
