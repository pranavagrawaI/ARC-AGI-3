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
    / "analysis/deployment-probes/20260902T-public1-multiaction-v12"
    / "EXPERIMENT-CARD.md"
)


def load_builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "public1_multiaction_v12_builder", BUILDER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def notebook_source(notebook: dict) -> str:
    return "\n".join("".join(cell.get("source", "")) for cell in notebook["cells"])


def test_candidate_notebook_is_deterministic_and_compiles() -> None:
    builder = load_builder()
    generated = builder.build()
    builder.validate(generated)
    assert generated == json.loads(NOTEBOOK.read_text())
    assert generated == builder.build()
    for index, cell in enumerate(generated["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"multiaction-cell-{index}", "exec")


def test_candidate_binds_multiaction_wheel_and_no_submission() -> None:
    builder = load_builder()
    notebook = builder.build()
    source = notebook_source(notebook)
    asset_source = "".join(notebook["cells"][1]["source"])
    metadata = json.loads(METADATA.read_text())

    assert "TYCHO_PUBLIC_GAME_IDS'] = 'ls20'" in source
    assert "TYCHO_KAGGLE_MAX_ACTIONS'] = '8'" in source
    assert "ARC_LLM_MAX_ACTOR_CALLS'] = '6'" in source
    assert "ARC_LLM_MAX_ACTOR_CALLS_AFTER_INITIAL'] = '1'" in source
    assert "ARC_LLM_MAX_TOKENS_AFTER_INITIAL'] = '16000'" in source
    assert "take_action and take_actions" in source
    assert "total_model_calls_avoided" in PROBE.read_text()
    assert base64.b64encode(RUNTIME.read_bytes()).decode("ascii") in asset_source
    assert base64.b64encode(PROBE.read_bytes()).decode("ascii") in asset_source
    assert (
        base64.b64encode(builder.LOCAL_SOLVER.read_bytes()).decode("ascii")
        in asset_source
    )
    assert (
        base64.b64encode(builder.ADAPTER.read_bytes()).decode("ascii") in asset_source
    )
    assert "kaggle competitions submit" not in source
    assert "KAGGLE_IS_COMPETITION_RERUN" in source

    assert sha256(RUNTIME) == builder.RUNTIME_SHA256
    assert sha256(PROBE) == builder.PROBE_SHA256
    assert (
        builder.WHEELHOUSE_MANIFEST_SHA256
        == "c24d0e1c1656fa6542b07a13ea1de4f6303efa4b5b5e078e2a5ec184ace67c8f"
    )
    assert (
        builder.SOLVER_WHEEL_SHA256
        == "1fa38bf6647338bfdacec771e1d6eb449c7d68a89ba98ea702367974cbe134ff"
    )
    assert "solver_manifest.get('artifact_revision') != 9" in source
    assert metadata["id"] == "pranavagrawai/prototype-tycho-nooa-public-3-full-timing"
    assert metadata["title"] == "PROTOTYPE Tycho NOOA public 3 full timing"
    title_slug = re.sub(r"[^a-z0-9]+", "-", metadata["title"].lower()).strip("-")
    assert metadata["id"].split("/", 1)[1] == title_slug
    assert metadata["is_private"] is True
    assert metadata["enable_internet"] is False
    assert metadata["machine_shape"] == "NvidiaRtxPro6000"
    assert metadata["competition_sources"] == ["arc-prize-2026-arc-agi-3"]
    assert (
        "pranavagrawai/tycho-nooa-solver-wheelhouse-cp312-v9"
        in metadata["dataset_sources"]
    )
    assert (
        "pranavagrawai/tycho-nooa-solver-wheelhouse-cp312-v8"
        not in metadata["dataset_sources"]
    )


def test_candidate_does_not_change_v11_control() -> None:
    control = HERE.parent / "s0-public3-full-timing-v11-action-horizon-candidate"
    assert (
        sha256(control / "s0_public3_full_timing.ipynb")
        == "4357651444d88664e00f2a241944139f3fa8155a8b5f9970febec5f4360f5ab5"
    )
    assert (
        sha256(control / "build_notebook.py")
        == "ee43ddf0a0f61bb7949bc7707c215407748b8c696ac4b39ece4f0f8dddce320f"
    )
    assert (
        sha256(control / "kernel-metadata.json")
        == "a759cfb3522893caf4e5c463071e66c8f032580de75dac85a6a016093b15f761"
    )


def test_card_requires_observed_batch_uptake_without_submission() -> None:
    text = CARD.read_text()
    assert "at least one accepted `take_actions` call" in text
    assert "at least one offset greater than zero" in text
    assert "Do not create a competition submission" in text
    assert "V11" in text and "16,000" in text
