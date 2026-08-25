#!/usr/bin/env python3
"""Preflight or execute the authorized short-then-long production shadow."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from tycho_nooa.llm import build_llm
from tycho_nooa.matched_actor_manifest import sha256_file
from tycho_nooa.production_shadow_launch import (
    execute_authorized_production_shadow,
    preflight_production_shadow,
)

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent


def _paths(phase: str) -> tuple[Path, Path]:
    root = PACKAGE / phase
    return root / "experiment.json", root / "authorization.json"


def _destination(manifest_path: Path, authorization_path: Path) -> Path:
    manifest = json.loads(manifest_path.read_text())
    authorization = json.loads(authorization_path.read_text())
    return ROOT / manifest["output_root"] / authorization["authorization_id"]


async def main(*, preflight_only: bool) -> None:
    llm, lane = build_llm()
    if lane != "live":
        raise RuntimeError("paid production shadow requires a live client")
    short_manifest, short_authorization = _paths("short")
    if preflight_only:
        manifest, authorization, authorization_hash = preflight_production_shadow(
            short_manifest,
            short_authorization,
            repository_root=ROOT,
            llm=llm,
        )
        print(
            json.dumps(
                {
                    "authorization_id": authorization.authorization_id,
                    "authorization_sha256": authorization_hash,
                    "experiment_manifest_sha256": sha256_file(short_manifest),
                    "maximum_calls": authorization.maximum_calls,
                    "maximum_spend_usd": str(authorization.maximum_spend_usd),
                    "phase": manifest.phase,
                    "status": "preflight_pass",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    short = await execute_authorized_production_shadow(
        short_manifest,
        short_authorization,
        repository_root=ROOT,
        destination=_destination(short_manifest, short_authorization),
        llm=llm,
    )
    print(json.dumps({"short": short}, indent=2, sort_keys=True))
    if short.get("advance") is not True:
        return

    long_manifest, long_authorization = _paths("long")
    long = await execute_authorized_production_shadow(
        long_manifest,
        long_authorization,
        repository_root=ROOT,
        destination=_destination(long_manifest, long_authorization),
        llm=llm,
    )
    print(json.dumps({"long": long}, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(preflight_only=args.preflight))
