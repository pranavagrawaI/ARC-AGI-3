"""Exact review identity, external donor bindings, and no mutable frozen artifacts."""

import hashlib
import json
from pathlib import Path

from bootstrap import HERE, RUNTIME, ROOT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def files():
    own = (
        list(HERE.glob("*.py"))
        + list(HERE.glob("*.md"))
        + [HERE / "operational-theory.schema.json", HERE / "PREFIX-BINDING.json"]
    )
    own += list((HERE / "evidence").glob("*")) + list((HERE / "metadata").glob("*"))
    donors = list((RUNTIME / "chassis").rglob("*.py")) + list(
        (RUNTIME / "chassis").rglob("*.md")
    )
    # Python dependencies actually defining the interaction and runtime boundary.
    site = ROOT / "deployment/kaggle/.venv/lib/python3.12/site-packages"
    donors += list((site / "nooa").rglob("*.py"))
    return sorted(set(own + donors))


def freeze():
    target = HERE / "MANIFEST.json"
    if target.exists():
        raise RuntimeError("freeze already exists; do not overwrite identity")
    manifest = {
        "task": "T1",
        "files": {str(p.relative_to(ROOT)): sha(p) for p in files()},
        "card_sha256": sha(HERE / "EXPERIMENT-CARD.md"),
        "provider_calls": 0,
    }
    target.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return sha(target)


def verify():
    target = HERE / "MANIFEST.json"
    manifest = json.loads(target.read_text())
    expected = manifest["files"]
    if set(expected) != {str(p.relative_to(ROOT)) for p in files()}:
        raise RuntimeError("manifest file set drift")
    for name, expected_sha in expected.items():
        path = ROOT / name
        if path.is_symlink() or not path.is_file() or sha(path) != expected_sha:
            raise RuntimeError("manifest drift: " + name)
    if sha(HERE / "EXPERIMENT-CARD.md") != manifest["card_sha256"]:
        raise RuntimeError("card drift")
    return sha(target)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    print(freeze() if parser.parse_args().freeze else verify())
