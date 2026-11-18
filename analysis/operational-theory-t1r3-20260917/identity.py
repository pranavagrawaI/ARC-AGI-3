"""Exact review identity, external donor bindings, and no mutable frozen artifacts."""

import hashlib
import json
from pathlib import Path

from bootstrap import HERE, RUNTIME, ROOT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def files():
    own = (
        [
            HERE / name
            for name in (
                "actor.py",
                "bootstrap.py",
                "client.py",
                "evidence_api.py",
                "evidence_authority.py",
                "test_evidence_integrity.py",
                "experiment.py",
                "fixtures.py",
                "helper_authority.py",
                "identity.py",
                "local_wire.py",
                "prefill.py",
                "protocol.py",
                "replay.py",
                "request_contract.py",
                "run_pair.py",
                "supervisor.py",
                "test_adversarial.py",
                "test_operational.py",
                "theory.py",
                "transport.py",
                "verify.py",
                "worker.py",
            )
        ]
        + [
            HERE / name
            for name in ("EXPERIMENT-CARD.md", "README.md", "workflow-skill.md")
        ]
        + [
            HERE / "operational-theory.schema.json",
            HERE / "PREFIX-BINDING.json",
            HERE / "PREDECESSOR.json",
            HERE / "reviewer-fixtures.json",
            HERE / "binding-review-fixture.json",
            HERE / "evidence-review-fixture.json",
        ]
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
        "task": "T1R3",
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
