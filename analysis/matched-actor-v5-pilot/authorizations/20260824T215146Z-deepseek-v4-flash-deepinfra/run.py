"""Preflight or execute one exact Matched Actor V5-Pilot authorization."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from nooa.unifiedllm import UnifiedLLM

from tycho_nooa.llm import build_llm
from tycho_nooa.matched_actor_manifest import sha256_file
from tycho_nooa.matched_actor_pilot_launch_v5 import (
    execute_authorized_matched_actor_pilot_v5,
    preflight_matched_actor_pilot_v5,
)

AUTHORIZATION_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
MANIFEST_PATH = AUTHORIZATION_ROOT / "experiment.json"
AUTHORIZATION_PATH = AUTHORIZATION_ROOT / "authorization.json"


def clients() -> tuple[UnifiedLLM, UnifiedLLM]:
    native_llm, native_lane = build_llm()
    codeact_llm, codeact_lane = build_llm()
    if (native_lane, codeact_lane) != ("live", "live"):
        raise RuntimeError("paid Matched Actor pilot requires two live clients")
    return native_llm, codeact_llm


async def main(*, preflight_only: bool) -> None:
    native_llm, codeact_llm = clients()
    if preflight_only:
        manifest, authorization, authorization_hash = (
            preflight_matched_actor_pilot_v5(
                MANIFEST_PATH,
                AUTHORIZATION_PATH,
                repository_root=REPOSITORY_ROOT,
                native_llm=native_llm,
                codeact_llm=codeact_llm,
            )
        )
        print(
            json.dumps(
                {
                    "authorization_id": authorization.authorization_id,
                    "authorization_sha256": authorization_hash,
                    "execution_authorized": manifest.execution_authorized,
                    "experiment_manifest_sha256": sha256_file(MANIFEST_PATH),
                    "maximum_calls": authorization.maximum_calls,
                    "maximum_spend_usd": str(authorization.maximum_spend_usd),
                    "output_root": manifest.output_root,
                    "status": "preflight_pass",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    manifest = json.loads(MANIFEST_PATH.read_text())
    authorization = json.loads(AUTHORIZATION_PATH.read_text())
    destination = (
        REPOSITORY_ROOT
        / manifest["output_root"]
        / authorization["authorization_id"]
    )
    await execute_authorized_matched_actor_pilot_v5(
        MANIFEST_PATH,
        AUTHORIZATION_PATH,
        repository_root=REPOSITORY_ROOT,
        destination=destination,
        native_llm=native_llm,
        codeact_llm=codeact_llm,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    arguments = parser.parse_args()
    asyncio.run(main(preflight_only=arguments.preflight))
