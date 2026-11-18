"""Package the selected dispatcher without importing changing loose worker files."""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = (
    HERE.parent
    / "s0-nooa-join-public3-reasoning-notebook/s0_nooa_join_public3_reasoning.ipynb"
)
DISPATCHER = (
    ROOT / "analysis/deployment-probes/20260905-request-ownership/candidate/headless.py"
)
BASE_SHA = "c4a7409a09a1a3a2a2033ae29ecfc86a13bf697d46ecea86b71b97b7bdca6cd8"
DISPATCHER_SHA = "a8d59eaf318edc97931cd503c329a56fb20d2891d82d288fec52986cceee2de1"
NOTEBOOK = HERE / "dispatcher_admission.ipynb"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def join_archive(notebook):
    found = []
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        for node in ast.walk(ast.parse(source)):
            if (
                not isinstance(node, ast.Constant)
                or not isinstance(node.value, str)
                or not node.value.startswith("UEsDB")
            ):
                continue
            data = base64.b64decode(node.value)
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                if "nooa_worker.py" in archive.namelist():
                    found.append((index, node.value, data))
    assert len(found) == 1, "expected exactly one embedded join archive"
    return found[0]


def build():
    assert sha(BASE.read_bytes()) == BASE_SHA
    assert sha(DISPATCHER.read_bytes()) == DISPATCHER_SHA
    baseline = json.loads(BASE.read_text())
    candidate = json.loads(BASE.read_text())
    index, encoded, original_zip = join_archive(baseline)
    buffer = io.BytesIO()
    with (
        zipfile.ZipFile(io.BytesIO(original_zip)) as old,
        zipfile.ZipFile(buffer, "w") as new,
    ):
        before = {name: sha(old.read(name)) for name in old.namelist()}
        for entry in old.infolist():
            data = (
                DISPATCHER.read_bytes()
                if entry.filename == "headless.py"
                else old.read(entry.filename)
            )
            # Avoid zlib-version-dependent bytes between local Python versions.
            entry.compress_type = zipfile.ZIP_STORED
            new.writestr(entry, data)
    zip_bytes = buffer.getvalue()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        after = {name: sha(archive.read(name)) for name in archive.namelist()}
    assert before.keys() == after.keys()
    assert [name for name in before if before[name] != after[name]] == ["headless.py"]
    source = "".join(candidate["cells"][index]["source"])
    assert source.count(encoded) == 1
    candidate["cells"][index]["source"] = source.replace(
        encoded, base64.b64encode(zip_bytes).decode()
    )
    candidate["cells"][0]["source"] = (
        "# Dispatcher admission candidate\n\n"
        "Only headless.py differs inside the frozen join archive. Local package qualification; "
        "interruption, supersession and target scheduler retirement remain unqualified. "
        "Inherited GPU/gameplay cells are not executed by local qualification.\n"
    )
    # Every engine, manifest, dependency, launch and reasoning cell stays exact.
    for i, (old, new) in enumerate(
        zip(baseline["cells"], candidate["cells"], strict=True)
    ):
        if i not in (0, index):
            assert old == new
    notebook_bytes = (
        json.dumps(candidate, indent=1, ensure_ascii=False) + "\n"
    ).encode()
    metadata = json.loads((BASE.parent / "kernel-metadata.json").read_text())
    metadata.update(
        id="pranavagrawai/prototype-nooa-dispatcher-admission",
        title="PROTOTYPE NOOA dispatcher admission",
        code_file=NOTEBOOK.name,
    )
    manifest = {
        "schema": "nooa-dispatcher-package-v1",
        "baseline_notebook_sha256": BASE_SHA,
        "notebook_sha256": sha(notebook_bytes),
        "join_zip_sha256": sha(zip_bytes),
        "changed_archive_members": ["headless.py"],
        "original_sources": before,
        "packaged_sources": after,
        "unchanged_cell_indices": [
            i for i in range(len(candidate["cells"])) if i not in (0, index)
        ],
        "scope": "dispatcher admission only; no engine/retry/reasoning/model changes",
    }
    return {
        NOTEBOOK: notebook_bytes,
        HERE / "join.zip": zip_bytes,
        HERE / "package-manifest.json": (
            json.dumps(manifest, indent=2) + "\n"
        ).encode(),
        HERE / "kernel-metadata.json": (json.dumps(metadata, indent=2) + "\n").encode(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for path, data in build().items():
        if args.check:
            assert path.read_bytes() == data, f"package drift: {path}"
        else:
            path.write_bytes(data)
    print(
        json.dumps(
            {"status": "verified" if args.check else "built", "notebook": str(NOTEBOOK)}
        )
    )


if __name__ == "__main__":
    main()
