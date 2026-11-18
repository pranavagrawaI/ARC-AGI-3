"""Build a separate private engine-only notebook from the frozen bootstrap."""

import argparse
import base64
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = (
    HERE.parent
    / "s0-nooa-join-public3-reasoning-notebook/s0_nooa_join_public3_reasoning.ipynb"
)
SLUG = "pranavagrawai/qwen38-isolated-engine-20260905-v1"


def build():
    assert (
        hashlib.sha256(BASE.read_bytes()).hexdigest()
        == "c4a7409a09a1a3a2a2033ae29ecfc86a13bf697d46ecea86b71b97b7bdca6cd8"
    )
    original = json.loads(BASE.read_text())
    notebook = json.loads(BASE.read_text())
    notebook["cells"][0]["source"] = (
        "# Isolated unchanged-target engine comparison\nNo Actor, games, dispatcher patch, submission or retries.\n"
    )
    # Asset cell is inert extraction; no worker is launched. Preserve frozen runtime and manifest bytes.
    assets = "".join(original["cells"][1]["source"])
    hashes = {}
    for name, target in [
        ("runner.py", "/tmp/engine-runner.py"),
        ("tuning.json", "/tmp/engine-tuning.json"),
    ]:
        data = (HERE / name).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        hashes[name] = digest
        assets += f"\n_payload = base64.b64decode({base64.b64encode(data).decode()!r})\nassert hashlib.sha256(_payload).hexdigest() == {digest!r}\nPath({target!r}).write_bytes(_payload)\n"
    notebook["cells"][1]["source"] = "import hashlib\n" + assets
    bootstrap = "".join(original["cells"][2]["source"])
    notebook["cells"][2]["source"] = bootstrap[
        : bootstrap.index("solver_manifest_path = unique_exact_file(")
    ]
    notebook["cells"][3]["source"] = (
        "import subprocess, sys\nfrom pathlib import Path\n"
        "out = Path('/kaggle/working/isolated-engine-20260905')\nout.mkdir(exist_ok=True)\n"
        "with (out/'runner.log').open('w') as log:\n"
        "    completed = subprocess.run([sys.executable, '/tmp/engine-runner.py'], stdout=log, stderr=subprocess.STDOUT, timeout=3300)\n"
        "print((out/'runner.log').read_text()[-12000:])\n"
        "assert completed.returncode == 0, f'engine probe failed: {completed.returncode}'\n"
    )
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
            compile("".join(cell["source"]), "<notebook>", "exec")
    metadata = json.loads((BASE.parent / "kernel-metadata.json").read_text())
    metadata.update(
        id=SLUG,
        title="Qwen38 isolated engine 20260905 v1",
        code_file="engine.ipynb",
        competition_sources=[],
    )
    metadata["dataset_sources"] = [
        x for x in metadata["dataset_sources"] if "solver-wheelhouse" not in x
    ]
    data = (json.dumps(notebook, indent=1) + "\n").encode()
    manifest = {
        "notebook_sha256": hashlib.sha256(data).hexdigest(),
        "assets": hashes,
        "baseline_notebook_sha256": hashlib.sha256(BASE.read_bytes()).hexdigest(),
        "slug": SLUG,
        "arms": ["baseline", "tuned-triton", "shallow-mtp"],
        "kernel_timeout_s": 3600,
        "runner_timeout_s": 3200,
    }
    return {
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
    print("Engine notebook verified" if args.check else "Engine notebook built")
