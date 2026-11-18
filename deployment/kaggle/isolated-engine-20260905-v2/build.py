"""Metadata-only successor: bind the unchanged engine notebook to ARC-AGI-3."""

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "isolated-engine-20260905"
SLUG = "pranavagrawai/qwen38-isolated-engine-20260905-v2"
COMPETITION = "arc-prize-2026-arc-agi-3"


def build():
    previous = json.loads((PREVIOUS / "manifest.json").read_text())
    notebook = (PREVIOUS / "engine.ipynb").read_bytes()
    assert hashlib.sha256(notebook).hexdigest() == previous["notebook_sha256"]
    metadata = json.loads((PREVIOUS / "kernel-metadata.json").read_text())
    metadata.update(
        id=SLUG,
        title="Qwen38 isolated engine 20260905 v2",
        competition_sources=[COMPETITION],
    )
    manifest = dict(
        previous,
        slug=SLUG,
        predecessor_manifest_sha256=hashlib.sha256(
            (PREVIOUS / "manifest.json").read_bytes()
        ).hexdigest(),
        competition_sources=[COMPETITION],
        scope="metadata only; notebook and embedded runner bytes unchanged",
    )
    return {
        "engine.ipynb": notebook,
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
