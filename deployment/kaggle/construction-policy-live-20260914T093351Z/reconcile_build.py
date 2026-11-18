#!/usr/bin/env python3
"""Reconcile the live package from the immutable reviewed r5 bytes."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ANALYSIS = REPO / "analysis" / HERE.name
R5 = HERE.parent / "construction-policy-integration-20260913T221412Z-r5"
R5A = REPO / "analysis/construction-policy-integration-20260913T221412Z-r5"
R5_HASHES = {
    "manifest": "493018986c25f14472fe6c10bd9c2c888cde4cafa8388d7425495687eb68d440",
    "notebook": "50d345ec00f2a5d089e19b4ee74412576f54a3902ba67859bd4bf9258d02dbd6",
    "build": "7135749d2a22bd89299a0306b7c6fb94a08efdba5eac1e6c5ba6e1dd66f2cf4d",
}
ASSETS = ["PROVIDER-SELECTION.json", "accelerator.py", "case.json", "constructor.py", "containment_base.py", "inspection_runner.py", "integration.py", "integration_boundary.py", "local_transport.py", "model-metadata.json", "qualified_runner.py", "real_trace.py", "request_contract.py", "remote_entry.py"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(write=True):
    assert sha(R5A / "MANIFEST.json") == R5_HASHES["manifest"]
    assert sha(R5 / "BUILD.json") == R5_HASHES["build"]
    assert sha(HERE / "construction.ipynb") == sha(R5 / "construction.ipynb") == R5_HASHES["notebook"]
    assert sha(HERE / "remote_entry.py") == sha(R5 / "remote_entry.py")
    for name in ASSETS:
        if name != "remote_entry.py":
            assert sha(ANALYSIS / name) == sha(R5A / name), name
    expected = json.loads((ANALYSIS / "SERVER-REFERENCE.json").read_text())
    assert json.loads((HERE / "kernel-metadata.json").read_text()) == expected
    build = {
        "identity": HERE.name,
        "baseline_identity": R5.name,
        "baseline_manifest_sha256": R5_HASHES["manifest"],
        "baseline_build_sha256": R5_HASHES["build"],
        "notebook_inherited_byte_identical": True,
        "assets": {name: sha(HERE / name if name == "remote_entry.py" else ANALYSIS / name) for name in ASSETS},
        "files": {"construction.ipynb": sha(HERE / "construction.ipynb"), "kernel-metadata.json": sha(HERE / "kernel-metadata.json")},
    }
    if write:
        (HERE / "BUILD.json").write_text(json.dumps(build, indent=2, sort_keys=True) + "\n")
    return build


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True))
