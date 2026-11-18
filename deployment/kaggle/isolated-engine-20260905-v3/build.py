"""Freeze the input-preparation correction; preserve all engine arms and caps."""

import argparse
import base64
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "isolated-engine-20260905-v2"
ORIGINAL = HERE.parent / "isolated-engine-20260905"
SLUG = "pranavagrawai/qwen38-isolated-engine-20260905-v3"
COMPETITION = "arc-prize-2026-arc-agi-3"


def build():
    manifest = json.loads((PREVIOUS / "manifest.json").read_text())
    old_runner = (ORIGINAL / "runner.py").read_bytes()
    assert hashlib.sha256(old_runner).hexdigest() == manifest["assets"]["runner.py"]
    source = old_runner.decode()
    start = source.index("    messages = [")
    end = source.index('    save(\n        OUT / "frozen-input.json",', start)
    source = (
        source[:start] + "    messages, ids = prepare_input(tokenizer)\n" + source[end:]
    )
    source = source.replace(
        "def main():",
        (HERE / "input_preparation.py").read_text() + "\n\ndef main():",
        1,
    )
    runner = source.encode()
    notebook = json.loads((PREVIOUS / "engine.ipynb").read_text())
    assert (
        hashlib.sha256((PREVIOUS / "engine.ipynb").read_bytes()).hexdigest()
        == manifest["notebook_sha256"]
    )
    old_b64 = base64.b64encode(old_runner).decode()
    assets = "".join(notebook["cells"][1]["source"])
    assert assets.count(old_b64) == 1
    assets = assets.replace(old_b64, base64.b64encode(runner).decode()).replace(
        manifest["assets"]["runner.py"], hashlib.sha256(runner).hexdigest()
    )
    notebook["cells"][1]["source"] = assets
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "<notebook>", "exec")
    compile(source, "runner.py", "exec")
    data = (json.dumps(notebook, indent=1) + "\n").encode()
    metadata = json.loads((PREVIOUS / "kernel-metadata.json").read_text())
    metadata.update(id=SLUG, title="Qwen38 isolated engine 20260905 v3")
    assert metadata["competition_sources"] == [COMPETITION]
    manifest.update(
        slug=SLUG,
        notebook_sha256=hashlib.sha256(data).hexdigest(),
        scope="explicit tokenizer return type and input validation only",
        predecessor_manifest_sha256=hashlib.sha256(
            (PREVIOUS / "manifest.json").read_bytes()
        ).hexdigest(),
    )
    manifest["assets"]["runner.py"] = hashlib.sha256(runner).hexdigest()
    return {
        "runner.py": runner,
        "engine.ipynb": data,
        "kernel-metadata.json": (json.dumps(metadata, indent=2) + "\n").encode(),
        "manifest.json": (json.dumps(manifest, indent=2) + "\n").encode(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for name, data in build().items():
        if args.check:
            assert (HERE / name).read_bytes() == data, name
        else:
            (HERE / name).write_bytes(data)
    print("Verified" if args.check else "Built")
