"""Bind the shared 0158 worker to the exact DeepSeek V4 Flash provider route."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ORIGINAL_RUNTIME = Path(__file__).resolve().parents[2] / "nooa-runtime"
sys.path.insert(0, str(ORIGINAL_RUNTIME))
spec = importlib.util.spec_from_file_location("_0158_shared_launcher", ORIGINAL_RUNTIME / "launcher.py")
if spec is None or spec.loader is None:
    raise RuntimeError("shared 0158 launcher is unavailable")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
launcher.LIVE_PROVIDER_ROUTE = {
    "order": ["coreweave/fp8"],
    "only": ["coreweave/fp8"],
    "allow_fallbacks": False,
    "require_parameters": False,
    "quantizations": ["fp8"],
}


if __name__ == "__main__":
    launcher.main()
