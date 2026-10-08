"""New bounded one-attempt run; reuse typed quota/priority/slot admission."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from build import HERE, SLUG, TAG
sys.path.insert(0, str(HERE.parent / "mtp-depth-20261006"))
import launch_watch


def verify():
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "."], cwd=HERE, text=True)
    if dirty:
        raise ValueError("uncommitted experiment: " + dirty)
    commit = subprocess.check_output(["git", "rev-parse", TAG + "^{commit}"], cwd=HERE, text=True).strip()
    if subprocess.check_output(["git", "diff", commit, "--", "."], cwd=HERE, text=True):
        raise ValueError("source differs from frozen tag")
    frozen = json.loads((HERE / "freeze.json").read_text())
    for name, digest in frozen["files"].items():
        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != digest:
            raise ValueError("hash mismatch: " + name)
    m = json.loads((HERE / "kernel/kernel-metadata.json").read_text())
    assert m["competition_sources"] == ["arc-prize-2026-arc-agi-3"]
    assert m["machine_shape"] == "NvidiaRtxPro6000" and m["is_private"] is True
    assert m["id"] == "pranavagrawai/" + SLUG
    # Original private dataset readback/source identity must still qualify.
    parent = launch_watch.verify_prelaunch.verify()
    return {"run_id": SLUG, "commit": commit, "freeze": frozen,
            "hard_timeout_s": 7200, "parent": parent}


if __name__ == "__main__":
    import requests
    send = requests.Session.send
    def bounded(session, request, **kwargs):
        kwargs["timeout"] = (15, 45)
        return send(session, request, **kwargs)
    requests.Session.send = bounded
    launch_watch.EVID = HERE / "evidence"
    launch_watch.NOTEBOOK_DIR = HERE / "kernel"
    launch_watch.HARD_TIMEOUT_S = 7200
    launch_watch.identity = verify
    launch_watch.main()
