"""Regenerate exact source-to-adaptation diffs and their content manifest."""

from __future__ import annotations

import json
import subprocess
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIRS = {
    "nooa-agent.patch": (
        "third_party/source-snapshots/nooa/solver_agent.py",
        "nooa-runtime/agent.py",
    ),
    "nooa-launcher.patch": (
        "third_party/source-snapshots/nooa/launcher.py",
        "nooa-runtime/launcher.py",
    ),
    "nooa-provider.patch": (
        "third_party/source-snapshots/arc-agi-3/0002-nooa-llm.py",
        "nooa-runtime/provider.py",
    ),
    "tycho-evidence.patch": (
        "third_party/source-snapshots/tycho/workspace.py",
        "host/evidence.py",
    ),
}


def diff_bytes(before: str, after: str) -> bytes:
    result = subprocess.run(
        ["diff", "-u", before, after], cwd=ROOT, capture_output=True, check=False
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    return result.stdout


def main() -> None:
    output = ROOT / "provenance"
    output.mkdir(exist_ok=True)
    records = []
    for name, (before, after) in PAIRS.items():
        payload = diff_bytes(before, after)
        (output / name).write_bytes(payload)
        records.append(
            {
                "patch": name,
                "before": before,
                "after": after,
                "sha256": sha256(payload).hexdigest(),
            }
        )
    (output / "adaptation-patches.json").write_text(
        json.dumps(
            {"schema": "0158-adaptation-patches-v1", "patches": records},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
