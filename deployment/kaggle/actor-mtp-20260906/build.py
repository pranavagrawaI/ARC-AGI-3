"""Build replay notebook; unsealed notebooks fail before installation or loading."""

import argparse
import base64
import hashlib
import json
import lzma
from pathlib import Path
from corpus import validate

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "isolated-engine-20260905-v3"
SLUG = "pranavagrawai/actor-mtp-20260906"


def build(corpus_path=None):
    old = json.loads((BASE / "manifest.json").read_text())
    assert (
        hashlib.sha256((BASE / "engine.ipynb").read_bytes()).hexdigest()
        == old["notebook_sha256"]
    )
    notebook = json.loads((BASE / "engine.ipynb").read_text())
    notebook["cells"][0]["source"] = (
        "# Actor request replay: baseline versus shallow MTP\nUser upload only. Read README and experiment card. No game/tool execution.\n"
    )
    value = json.loads(corpus_path.read_text()) if corpus_path else None
    digest = validate(value) if value is not None else None
    # Kaggle rejects kernel source >=1 MiB. The 14 captures share a growing
    # message history, so LZMA takes 1.6 MB to ~24 KB. Sealing is unchanged:
    # the same object is digested and validated after decompression.
    encoded = base64.b64encode(
        lzma.compress(json.dumps(value).encode(), preset=9 | lzma.PRESET_EXTREME)
    ).decode()
    contract = (HERE / "corpus.py").read_text()
    preflight = (
        "import base64, json, hashlib, lzma\nfrom pathlib import Path\n" + contract + "\n"
        f"corpus = json.loads(lzma.decompress(base64.b64decode({encoded!r})))\n"
        "if corpus is None:\n    raise RuntimeError('NOT SEALED: full Actor requests missing. Do not spend a GPU run; rebuild with --corpus.')\n"
        f"assert validate(corpus) == {digest!r}\n"
        "Path('/tmp/actor-corpus.json').write_text(json.dumps(corpus))\n"
        "import subprocess\n_gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).strip()\n"
        "assert 'RTX PRO 6000' in _gpu, _gpu\n"
    )
    assets = "".join(notebook["cells"][1]["source"])
    hashes = {}
    for name, target in [
        ("runner.py", "/tmp/actor-replay.py"),
        ("corpus.py", "/tmp/actor-corpus-contract.py"),
    ]:
        data = (HERE / name).read_bytes()
        hashes[name] = hashlib.sha256(data).hexdigest()
        assets += f"\n_data=base64.b64decode({base64.b64encode(data).decode()!r})\nassert hashlib.sha256(_data).hexdigest()=={hashes[name]!r}\nPath({target!r}).write_bytes(_data)\n"
    notebook["cells"][1]["source"] = assets
    notebook["cells"][3]["source"] = (
        "import subprocess, sys\nfrom pathlib import Path\n"
        "out=Path('/kaggle/working/actor-mtp-20260906'); out.mkdir(exist_ok=True)\n"
        "with (out/'runner.log').open('w') as log:\n"
        "    completed=subprocess.run([sys.executable,'/tmp/actor-replay.py'],stdout=log,stderr=subprocess.STDOUT,timeout=6600)\n"
        "print((out/'runner.log').read_text()[-12000:])\nassert completed.returncode==0\n"
    )
    notebook["cells"].insert(
        1,
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": preflight,
        },
    )
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "<notebook>", "exec")
    data = (json.dumps(notebook, indent=1) + "\n").encode()
    metadata = json.loads((BASE / "kernel-metadata.json").read_text())
    metadata.update(
        id=SLUG, title="Actor MTP 20260906", code_file="actor_mtp.ipynb"
    )
    manifest = {
        "schema": "actor-mtp-notebook-v1",
        "upload_ready": value is not None,
        "corpus_sha256": digest,
        "assets": hashes,
        "notebook_sha256": hashlib.sha256(data).hexdigest(),
        "session_cap_s": 7200,
        "runner_cap_s": 6400,
        "baseline_manifest_sha256": hashlib.sha256(
            (BASE / "manifest.json").read_bytes()
        ).hexdigest(),
    }
    return {
        "actor_mtp.ipynb": data,
        "kernel-metadata.json": (json.dumps(metadata, indent=2) + "\n").encode(),
        "manifest.json": (json.dumps(manifest, indent=2) + "\n").encode(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for name, data in build(args.corpus).items():
        if args.check:
            assert (HERE / name).read_bytes() == data, name
        else:
            (HERE / name).write_bytes(data)
    print(
        "READY" if args.corpus else "NOT SEALED: request corpus required before upload"
    )
