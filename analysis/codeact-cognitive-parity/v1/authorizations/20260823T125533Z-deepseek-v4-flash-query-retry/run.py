"""Execute the precommitted cognitive-parity screen; configuration is shell-owned."""

from __future__ import annotations

import asyncio
from pathlib import Path

from tycho_nooa.llm import build_llm
from tycho_nooa.parity_manifest import load_experiment_manifest
from tycho_nooa.parity_screen import execute_authorized_screen

AUTHORIZATION_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
MANIFEST_PATH = AUTHORIZATION_ROOT / "experiment.json"


async def main() -> None:
    manifest = load_experiment_manifest(MANIFEST_PATH, repository_root=REPOSITORY_ROOT)
    manifest.require_execution_authorized()
    native_llm, native_lane = build_llm()
    codeact_llm, codeact_lane = build_llm()
    if (native_lane, codeact_lane) != ("live", "live"):
        raise RuntimeError("authorized screen requires two live model clients")
    await execute_authorized_screen(
        MANIFEST_PATH,
        repository_root=REPOSITORY_ROOT,
        destination=REPOSITORY_ROOT / manifest.output_root,
        native_llm=native_llm,
        codeact_llm=codeact_llm,
    )


if __name__ == "__main__":
    asyncio.run(main())
