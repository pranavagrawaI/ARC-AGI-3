#!/usr/bin/env python3
"""Refuse Kaggle push until coordinator OpenRouter PASS. Never calls the provider from tests."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PASS = HERE / "OPENROUTER-PASS.json"


def main() -> int:
    if not PASS.is_file():
        print(json.dumps({"pushed": False, "reason": "OpenRouter PASS artifact absent"}))
        return 2
    payload = json.loads(PASS.read_text())
    if payload.get("status") != "PASS":
        print(json.dumps({"pushed": False, "reason": "OpenRouter PASS status is not PASS"}))
        return 2
    print(json.dumps({"pushed": False, "reason": "launch remains coordinator-owned even with PASS"}))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
