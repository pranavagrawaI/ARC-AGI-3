from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType

HERE = Path(__file__).resolve().parents[1]
BUILDER = HERE / "build_notebook.py"
NOTEBOOK = HERE / "s0_public3_full_timing.ipynb"
METADATA = HERE / "kernel-metadata.json"
PROBE = HERE / "public25_launcher_probe.py"
RUNTIME = HERE / "exact_request_outer_runtime.py"
CARD = (
    HERE.parents[2]
    / "analysis/deployment-probes/20260901T-public3-full-timing-v11-action-horizon"
    / "EXPERIMENT-CARD.md"
)


def load_builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "full_timing_v11_action_horizon_builder", BUILDER
    )
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
    local_solver_source = builder.LOCAL_SOLVER.read_text()
    assert "whole_model_request_count_verified" in runtime_source
    assert "minimum_launcher_solver_requests" not in runtime_source
    assert (
        "_MODEL_GENERATION_ADMISSION = threading.BoundedSemaphore(1)"
        in local_solver_source
    )
    assert base64.b64encode(RUNTIME.read_bytes()).decode("ascii") in asset_source
    assert base64.b64encode(PROBE.read_bytes()).decode("ascii") in asset_source
    assert (
        base64.b64encode(builder.LOCAL_SOLVER.read_bytes()).decode("ascii")
        in asset_source
    )
    assert (
        base64.b64encode(builder.ADAPTER.read_bytes()).decode("ascii")
        in asset_source
    )
    assert "kaggle competitions submit" not in source
    assert "KAGGLE_IS_COMPETITION_RERUN" in source

    assert sha256(RUNTIME) == builder.RUNTIME_SHA256
    assert sha256(PROBE) == builder.PROBE_SHA256
    assert metadata["id"] == (
        "pranavagrawai/prototype-tycho-nooa-public-3-full-timing"
    )
    assert metadata["title"] == "PROTOTYPE Tycho NOOA public 3 full timing"
    title_slug = re.sub(r"[^a-z0-9]+", "-", metadata["title"].lower()).strip("-")
    assert metadata["id"].split("/", 1)[1] == title_slug
    assert builder.WHEELHOUSE_MANIFEST_SHA256 == (
        "e95b8176e9b445059f858d511559017bcc6452eba96c1cfd4805e5c1d1d2c187"
    )
    assert builder.SOLVER_WHEEL_SHA256 == (
        "0ba1282ea11b679524fce1f813da21c12132f3a9b5a48e66eba4034fbc8f1322"
    )
    assert builder.sha256(builder.ADAPTER) == (
        "b0e4e2c94cd895eb7cdb5ebde78df6ff00220ed92ee88d510f5d2692da2d5a8f"
    )
    assert builder.sha256(builder.LOCAL_SOLVER) == (
        "517572979803525af52e7a6f2066fdee2eb3ce5923fd2780c5b185cb52b06795"
    )
    assert metadata["code_file"] == NOTEBOOK.name
    assert metadata["is_private"] is True
    assert metadata["enable_internet"] is False
    assert metadata["machine_shape"] == "NvidiaRtxPro6000"
    assert metadata["competition_sources"] == ["arc-prize-2026-arc-agi-3"]
    assert "pranavagrawai/tycho-nooa-solver-wheelhouse-cp312-v8" in metadata[
        "dataset_sources"
    ]
    assert "pranavagrawai/tycho-nooa-solver-wheelhouse-cp312-v7" not in metadata[
        "dataset_sources"
    ]
    assert "solver_manifest.get('artifact_revision') != 8" in source
    assert "'dataset_version': 1" in source


def test_candidate_does_not_change_running_v10_artifacts() -> None:
    running = HERE.parent / "s0-public3-full-timing-v10-candidate"

    assert sha256(running / "s0_public3_full_timing.ipynb") == (
        "c973761c06340edeb16324852eafa57f98a1d021a5fc5fd8af72a8346bfdecbb"
    )
    assert sha256(running / "build_notebook.py") == (
        "33da523fc5349b64a0b3adc086bd2fd43e16af1718867b0a0734b5176b2b8e29"
    )
    assert sha256(running / "kernel-metadata.json") == (
        "6a9517e6eff6ec5da795bd4e1fcedb049e9605cd4859b7d5fb44cabf879edd3c"
    )


def test_card_binds_v10_as_horizonless_predecessor_and_forbids_early_push() -> None:
    text = CARD.read_text()

    assert "V10 is the horizonless predecessor" in text
    assert "Do not push V11" in text
    assert "V10 terminal output and source bytes are preserved" in text
