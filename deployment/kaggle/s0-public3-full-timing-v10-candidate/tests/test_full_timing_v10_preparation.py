from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

HERE = Path(__file__).resolve().parents[1]
BUILDER = HERE / "build_notebook.py"
NOTEBOOK = HERE / "s0_public3_full_timing.ipynb"
METADATA = HERE / "kernel-metadata.json"
PROBE = HERE / "public25_launcher_probe.py"
RUNTIME = HERE / "exact_request_outer_runtime.py"


def load_builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location("full_timing_v10_builder", BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def notebook_source(notebook: dict) -> str:
    return "\n".join(
        "".join(cell.get("source", "")) for cell in notebook["cells"]
    )


def test_candidate_full_timing_notebook_is_deterministic_and_compiles() -> None:
    builder = load_builder()
    generated = builder.build()
    builder.validate(generated)

    assert generated == json.loads(NOTEBOOK.read_text())
    assert generated == builder.build()
    for index, cell in enumerate(generated["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"full-timing-cell-{index}", "exec")


def test_candidate_binds_full_horizon_exact_requests_and_no_submission() -> None:
    builder = load_builder()
    notebook = builder.build()
    source = notebook_source(notebook)
    asset_source = "".join(notebook["cells"][1]["source"])
    metadata = json.loads(METADATA.read_text())

    assert "TYCHO_PUBLIC_GAME_IDS'] = 'ls20,s5i5,sb26'" in source
    assert "TYCHO_KAGGLE_MAX_ACTIONS'] = '32'" in source
    assert "ARC_LLM_MAX_ACTOR_CALLS'] = '6'" in source
    assert "ARC_LLM_MAX_ACTOR_CALLS_AFTER_INITIAL'] = '1'" in source
    assert "ARC_LLM_MAX_TOKENS'] = '16000'" in source
    assert "ARC_LLM_MAX_TOKENS_AFTER_INITIAL'] = '16000'" in source
    assert "ARC_LLM_FORCE_ACTION_TOOL'] = '1'" in source
    assert "ARC_LLM_RETRIES'] = '0'" in source
    runtime_source = RUNTIME.read_text()
    assert "whole_model_request_count_verified" in runtime_source
    assert "minimum_launcher_solver_requests" not in runtime_source
    assert base64.b64encode(RUNTIME.read_bytes()).decode("ascii") in asset_source
    assert base64.b64encode(PROBE.read_bytes()).decode("ascii") in asset_source
    assert "kaggle competitions submit" not in source
    assert "KAGGLE_IS_COMPETITION_RERUN" in source

    assert sha256(RUNTIME) == builder.RUNTIME_SHA256
    assert sha256(PROBE) == builder.PROBE_SHA256
    assert metadata["id"] == (
        "pranavagrawai/prototype-tycho-nooa-public-3-continued-play"
    )
    assert metadata["code_file"] == NOTEBOOK.name
    assert metadata["is_private"] is True
    assert metadata["enable_internet"] is False
    assert metadata["machine_shape"] == "NvidiaRtxPro6000"
    assert metadata["competition_sources"] == ["arc-prize-2026-arc-agi-3"]


def test_candidate_does_not_change_v9_artifacts() -> None:
    running = HERE.parent / "s0-public3-successor-budget-v9-candidate"

    assert sha256(running / "s0_public3_continued_play.ipynb") == (
        "e64f731a9a3f9abe1cef9ba54396d005f08d01e6f90caef0d924248314910fbf"
    )
    assert sha256(running / "build_notebook.py") == (
        "16e4c0db66e7b624206d3e652b1b6a1f826730b725bd9778bad18c699b6e832f"
    )
    assert sha256(running / "kernel-metadata.json") == (
        "3e33ce63f2359298ac4a010f5bd9e81c5036a1c201e63efb42d0fc0b5fdeed1c"
    )
