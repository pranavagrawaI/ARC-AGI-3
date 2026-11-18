#!/usr/bin/env python3
"""Executable q6 acceptance suite; run with the reference NOOA venv Python."""

from pathlib import Path
import pytest

if __name__ == "__main__":
    raise SystemExit(
        pytest.main([str(Path(__file__).with_name("test_q6_contract.py")), "-q"])
    )
