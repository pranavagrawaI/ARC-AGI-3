"""Coordinator-only one-shot OpenRouter launch, bound to reviewed sources.

No credentials are loaded here; the reviewed harness owns its private relay key.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    run = Path(sys.argv[1]).resolve()
    manifest_path = run / "PRELAUNCH-MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    review = json.loads((run / "REVIEW.json").read_text())
    if not isinstance(review, dict) or review.get("status") != "APPROVED" or review.get("manifest_sha256") != digest(manifest_path):
        raise RuntimeError("Missing affirmative review for exact manifest")
    if not isinstance(manifest, dict) or not manifest.get("sources"):
        raise RuntimeError("Empty manifest")
    changed = [p for p, sha in manifest["sources"].items() if digest(ROOT / p) != sha]
    if changed:
        raise RuntimeError(f"Source drift before launch: {changed}")
    expected = ["unshare", "--user", "--map-auto", "--map-root-user", str(ROOT / "deployment/kaggle/.venv/bin/python"), str(ROOT / "analysis/actor-critic-construction-live-launch/harness.py"), "--execute", "--out", str(run / "raw")]
    if manifest.get("command") != expected or manifest.get("run_dir") != str(run):
        raise RuntimeError("Wrong launch command or identity")
    if (run / "raw").exists():
        raise RuntimeError("Raw directory already exists")
    with (run / "LAUNCH.json").open("x") as handle:
        json.dump({"manifest_sha256": digest(manifest_path), "command": expected}, handle, indent=2)
    with (run / "console.log").open("x") as output:
        completed = subprocess.run(expected, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, check=False)
    changed = [p for p, sha in manifest["sources"].items() if digest(ROOT / p) != sha]
    result = {"returncode": completed.returncode, "source_drift": changed, "manifest_sha256": digest(manifest_path)}
    (run / "COORDINATOR-EXECUTION.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return 1 if changed else completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
