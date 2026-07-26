"""THROWAWAY treatment-member launcher; reuses the 0002 runner unchanged."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
AGENTS_ROOT = HERE.parents[1]
BASELINE_ROOT = AGENTS_ROOT / "0002-llm-repl-agent"
REPO = AGENTS_ROOT.parents[1]
for path in (REPO, BASELINE_ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from prototype_agent import PrototypeCapabilityAgent


def _load_baseline_runner() -> Any:
    path = BASELINE_ROOT / "run.py"
    spec = importlib.util.spec_from_file_location("prototype_baseline_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load baseline runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    runner = _load_baseline_runner()
    runner.LlmReplAgent = PrototypeCapabilityAgent
    original_write_json = runner._write_json

    def write_json(path: Path, value: Any) -> None:
        if path.name == "manifest.json" and isinstance(value, dict):
            value = {
                **value,
                "agent": "0006-crude-capability-prototype",
                "prototype": {
                    "schema": "crude-0006-potency-v0",
                    "status": "throwaway_nonconformant",
                    "intervention": [
                        "exact_last_frame_delta",
                        "persistent_model_authored_hypotheses",
                        "pre_action_prediction",
                        "contradiction_check",
                        "probe_goal",
                    ],
                },
            }
        original_write_json(path, value)

    runner._write_json = write_json
    runner.main()


if __name__ == "__main__":
    main()
