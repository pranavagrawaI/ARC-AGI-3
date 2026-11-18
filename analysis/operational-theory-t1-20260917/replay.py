"""Replay canonical decisions from standalone arm artifacts, without model/transcript."""

import argparse
import json
from pathlib import Path

from experiment import catalog
from theory import replay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arm_directory", type=Path)
    args = parser.parse_args()
    rows = json.loads((args.arm_directory / "EVIDENCE.json").read_text())
    print(
        json.dumps(
            replay(args.arm_directory / "canonical", catalog(rows)),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
