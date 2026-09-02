#!/usr/bin/env python3
"""Build the deterministic predecessor-output mount qualification notebook."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def encoded(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def build_notebook() -> dict[str, object]:
    probe = encoded(HERE / "predecessor_mount_probe.py")
    fixture = encoded(HERE / "checkpoint-fixture.json")
    source = f"""import base64
import subprocess
import sys
from pathlib import Path

PROBE_B64 = {probe!r}
FIXTURE_B64 = {fixture!r}

probe_path = Path('/tmp/tycho_nooa_predecessor_mount_probe.py')
fixture_path = Path('/tmp/tycho_nooa_predecessor_checkpoint_fixture.json')
probe_path.write_bytes(base64.b64decode(PROBE_B64))
fixture_path.write_bytes(base64.b64decode(FIXTURE_B64))
output_root = Path('/kaggle/working/tycho-nooa-predecessor-mount')
command = [
    sys.executable,
    str(probe_path),
    '--input-root',
    '/kaggle/input',
    '--output-root',
    str(output_root),
    '--fixture',
    str(fixture_path),
]
subprocess.run(command, check=True)
print((output_root / 'result.json').read_text())
"""
    return {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source,
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "c0_predecessor_mount.ipynb",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_notebook(), indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
