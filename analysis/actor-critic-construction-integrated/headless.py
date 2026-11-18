#!/usr/bin/env python3
"""Launch v11 headless with ConstructionAwareAgent. Incumbent headless.py is not edited.

Constructor config is CLI on this process (whitelisted NOOA_AGENT_COMMAND), not worker env.
Remaining argv is forwarded to v11 parse_args.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V11 = ROOT / "deployment/kaggle/submission-runtime-20260912-v11"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(V11))
sys.path.insert(0, str(V11 / "chassis"))

spec = importlib.util.spec_from_file_location("v11_headless", V11 / "headless.py")
v11 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(v11)

from construction_agent import ConstructionAwareAgent  # noqa: E402  # chassis path required before this import

v11.MdArcSolverAgent = ConstructionAwareAgent


def apply_constructor_argv(argv: list[str]) -> list[str]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--constructor-transport")
    parser.add_argument("--constructor-fixture")
    parser.add_argument("--constructor-url")
    args, rest = parser.parse_known_args(argv)
    if args.constructor_transport:
        os.environ["ARC_CONSTRUCTOR_TRANSPORT"] = args.constructor_transport
    if args.constructor_fixture:
        os.environ["ARC_CONSTRUCTOR_FIXTURE"] = args.constructor_fixture
    if args.constructor_url:
        os.environ["ARC_CONSTRUCTOR_URL"] = args.constructor_url
    return rest


if __name__ == "__main__":
    if os.environ.get("ARC_LLM_MODEL"):
        os.environ.setdefault("ARC_LLM_MODEL", os.environ["ARC_LLM_MODEL"])
    sys.argv = [sys.argv[0], *apply_constructor_argv(sys.argv[1:])]
    asyncio.run(v11.run())
