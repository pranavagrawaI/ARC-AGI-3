#!/usr/bin/env python3
"""Run the generated notebook's fixture branches with isolated local roots."""
import argparse
import json
import os
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tests"))
from request_contract import LOCAL_MODEL
from test_integration import CANDIDATE, WORK
from test_local_wire import ScriptedServer, local_response


def tool(identifier, name, args):
    return {"id": identifier, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("normal", "cache-poison", "local-http"))
    parser.add_argument("notebook", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=False)
    (args.output_root / "tmp").mkdir()
    notebook = json.loads(args.notebook.read_text())
    source = notebook["cells"][1]["source"]
    if isinstance(source, list):
        source = "".join(source)
    source = source.replace("/tmp/construction-policy-integration", str(args.output_root / "tmp/construction-policy-integration"))
    source = source.replace("/kaggle/working", str(args.output_root / "working"))
    notebook["cells"][1]["source"] = source
    localized = args.output_root / "construction.localized.ipynb"
    localized.write_text(json.dumps(notebook, separators=(",", ":")) + "\n")

    server = None
    os.environ["CONSTRUCTION_LOCAL_TIMEOUT_S"] = "900"
    if args.mode == "local-http":
        responses = [
            local_response(tool("notebook.inspect/a", "execute_python", {"code": "import json;print(len(json.load(open('/evidence/case.json'))['visible_transitions']))"})),
            local_response(tool("notebook.inspect/b", "execute_python", {"code": "import json;print(json.load(open('/evidence/case.json'))['frontier'])"})),
            local_response(tool("notebook.submit/c", "submit_candidate", {"work_item": WORK, "candidate_source": CANDIDATE})),
        ]
        server = ScriptedServer(responses)
        os.environ.pop("CONSTRUCTION_FIXTURE", None)
        os.environ["CONSTRUCTION_LOCAL_BASE_URL"] = server.base
        os.environ["CONSTRUCTION_LOCAL_TIMEOUT_S"] = "900"
    else:
        os.environ["CONSTRUCTION_FIXTURE"] = "1"
        os.environ.pop("CONSTRUCTION_LOCAL_BASE_URL", None)
    if args.mode == "cache-poison":
        for name in ("accelerator", "constructor", "containment_base", "integration", "integration_boundary", "local_transport", "real_trace", "request_contract"):
            poisoned = types.ModuleType(name)
            poisoned.__file__ = "/tmp/cache-poison.py"
            sys.modules[name] = poisoned
    try:
        try:
            exec(compile(source, str(localized), "exec"), {"__name__": "__main__"})
        except SystemExit as error:
            if error.code not in (None, 0):
                raise
        if server is not None:
            (args.output_root / "working/notebook-local-wire.json").write_text(json.dumps([request.decode() for request in server.requests], indent=2) + "\n")
    finally:
        if server is not None:
            server.close()


if __name__ == "__main__":
    main()
