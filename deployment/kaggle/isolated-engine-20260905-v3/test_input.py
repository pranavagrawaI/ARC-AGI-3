"""Exercise the delivered runner with hash-verified checkpoint tokenizer files."""

import ast
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import transformers
from transformers import AutoTokenizer, BatchEncoding
from build import HERE, build

ROOT = HERE.parents[2]
OUT = ROOT / "analysis/deployment-probes/20260905-isolated-engine-v3"


def main():
    assert transformers.__version__ == "5.12.1"
    for name, data in build().items():
        assert (HERE / name).read_bytes() == data, name
    checks = json.loads(
        (
            ROOT
            / "analysis/deployment-probes/20260905-isolated-engine-v2/kaggle-output/isolated-engine-20260905/model-verification.json"
        ).read_text()
    )["files"]
    hashes = {}
    for path in (OUT / "tokenizer").iterdir():
        if path.suffix not in (".json", ".jinja"):
            continue
        record = next(r for r in checks if r["path"] == path.name)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == record["actual_sha256"], record
        hashes[path.name] = actual
    tokenizer = AutoTokenizer.from_pretrained(
        OUT / "tokenizer", local_files_only=True, trust_remote_code=True
    )
    spec = importlib.util.spec_from_file_location(
        "candidate_runner", HERE / "runner.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    notebook = json.loads((HERE / "engine.ipynb").read_text())
    strings = [
        n.value
        for n in ast.walk(ast.parse("".join(notebook["cells"][1]["source"])))
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    ]
    assert base64.b64encode((HERE / "runner.py").read_bytes()).decode() in strings
    messages, ids = runner.prepare_input(tokenizer)
    old = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=True
    )
    assert isinstance(old, BatchEncoding)
    try:
        json.dumps({"input_ids": old})
    except TypeError as error:
        reproduced = str(error)
    else:
        raise AssertionError("Old failure did not reproduce")
    assert ids == old["input_ids"]
    runner.save(
        OUT / "local-frozen-input.json", {"messages": messages, "input_ids": ids}
    )
    source = ast.parse((HERE / "runner.py").read_text())
    # Evaluate the actual runner's request literal with each scheduled limit.
    payload_expr = next(
        n.value
        for n in ast.walk(source)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "payload" for t in n.targets)
    )
    payloads = []
    for limit, ignore in [(64, True), (1024, True), (16000, False)]:
        payload = eval(
            compile(ast.Expression(payload_expr), "<actual-request>", "eval"),
            {"ids": ids, "limit": limit, "ignore": ignore},
        )
        assert json.loads(json.dumps(payload)) == payload
        assert payload["input_ids"] == ids
        payloads.append(payload)
    runner.save(OUT / "local-request-payloads.json", payloads)
    result = {
        "status": "PASS",
        "transformers": transformers.__version__,
        "old_failure": reproduced,
        "token_count": len(ids),
        "same_token_ids": True,
        "packaged_runner_tested": True,
        "tokenizer_sha256": hashes,
        "full_request_json_roundtrip": True,
    }
    runner.save(OUT / "LOCAL-TEST.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
