"""Pinned tokenizer smoke for all captured inputs, not SGLang rendering parity."""

import hashlib
import json
from pathlib import Path
import transformers
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    assert transformers.__version__ == "5.12.1"
    folder = ROOT / "analysis/deployment-probes/20260905-isolated-engine-v3/tokenizer"
    verified = json.loads(
        (
            ROOT
            / "analysis/deployment-probes/20260905-isolated-engine-v2/kaggle-output/isolated-engine-20260905/model-verification.json"
        ).read_text()
    )["files"]
    for path in folder.iterdir():
        if path.suffix not in (".json", ".jinja"):
            continue
        record = next(r for r in verified if r["path"] == path.name)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["actual_sha256"]
    tokenizer = AutoTokenizer.from_pretrained(
        folder, local_files_only=True, trust_remote_code=True
    )
    value = json.loads((HERE / "corpus.json").read_text())
    counts = []
    for record in value["records"]:
        request = record["request"]
        # HTTP bodies contain JSON-string arguments. SGLang serving_chat.py's
        # normalize_assistant_tool_call_arguments converts these for the template.
        # Do this on a test-only copy; never rewrite the transmitted corpus.
        messages = json.loads(json.dumps(request["messages"]))
        for message in messages:
            if message["role"] == "assistant":
                for call in message.get("tool_calls") or []:
                    arguments = call["function"].get("arguments")
                    if isinstance(arguments, str):
                        call["function"]["arguments"] = json.loads(arguments)
        ids = tokenizer.apply_chat_template(
            messages,
            tools=request["tools"],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            **request.get("chat_template_kwargs", {}),
        )
        assert isinstance(ids, list) and ids and all(type(t) is int for t in ids)
        assert json.loads(json.dumps({"input_ids": ids}))["input_ids"] == ids
        counts.append(
            {
                "id": record["id"],
                "template_tokens": len(ids),
                "captured_server_tokens": record["captured_usage"]["prompt_tokens"],
            }
        )
    print(
        json.dumps(
            {
                "status": "PASS",
                "scope": "Template-only smoke with verified tokenizer files; runtime replay uses server-rendered IDs",
                "cases": counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
