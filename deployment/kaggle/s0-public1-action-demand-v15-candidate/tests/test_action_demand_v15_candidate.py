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
    / "analysis/deployment-probes/20260903T-public1-action-demand-v15"
    / "EXPERIMENT-CARD.md"
)


def load_builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "public1_action_demand_v15_builder", BUILDER
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
            compile("".join(cell["source"]), f"action-demand-cell-{index}", "exec")


def test_candidate_binds_bounded_action_demand_wheel_and_no_submission() -> None:
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
    assert "ARC_LLM_ACTION_DEMAND_MAX_TOKENS'] = '2048'" in source
    assert "ARC_LLM_ACTION_DEMAND_DISABLE_THINKING'] = '1'" in source
    assert "exposes only" in source and "take_actions" in source
    assert "take_action and take_actions" not in source
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

    # V15 repair: local_solver must forward the action-demand knobs, and the
    # probe must expect them in the recorded decision envelope.
    local_solver_source = builder.LOCAL_SOLVER.read_text()
    assert '"ARC_LLM_ACTION_DEMAND_MAX_TOKENS"' in local_solver_source
    assert '"ARC_LLM_ACTION_DEMAND_DISABLE_THINKING"' in local_solver_source
    probe_source = PROBE.read_text()
    assert '"action_demand_max_output_tokens": int(' in probe_source
    assert '"action_demand_disable_thinking": (' in probe_source
    assert (
        builder.WHEELHOUSE_MANIFEST_SHA256
        == "518f8407b0ab1234430fc0c167e2ecd62ce70318a62292f8397e1a7ea4ed3862"
    )
    assert (
        builder.SOLVER_WHEEL_SHA256
        == "9ffadfc88ed40667fa14b876b5a17b94ac4992ca277377c887de0db5b0df7faf"
    )
    assert "solver_manifest.get('artifact_revision') != 11" in source
    assert metadata["id"] == "pranavagrawai/prototype-tycho-nooa-public-3-full-timing"
    assert metadata["title"] == "PROTOTYPE Tycho NOOA public 3 full timing"
    title_slug = re.sub(r"[^a-z0-9]+", "-", metadata["title"].lower()).strip("-")
    assert metadata["id"].split("/", 1)[1] == title_slug
    assert metadata["is_private"] is True
    assert metadata["enable_internet"] is False
    assert metadata["machine_shape"] == "NvidiaRtxPro6000"
    assert metadata["competition_sources"] == ["arc-prize-2026-arc-agi-3"]
    assert (
        "pranavagrawai/tycho-nooa-solver-wheelhouse-cp312-v11"
        in metadata["dataset_sources"]
    )
    assert (
        "pranavagrawai/tycho-nooa-solver-wheelhouse-cp312-v10"
        not in metadata["dataset_sources"]
    )


def test_candidate_does_not_change_v13_control() -> None:
    control = HERE.parent / "s0-public1-singular-action-v13-candidate"
    assert (
        sha256(control / "s0_public3_full_timing.ipynb")
        == "d4d4e948053ea582c67a8ac1af9544dee339790755f9ebee7d4b464f6132c096"
    )
    assert (
        sha256(control / "build_notebook.py")
        == "b3f9078f75f7d8f9cb2c3449b2360f7102dd1a7687bd8c462eb540fd3178b004"
    )
    assert (
        sha256(control / "kernel-metadata.json")
        == "5a43e7f3c5d9f772f137dc49e12c7970dd3b1006f608e91c6411d65115b0b2d0"
    )


def test_card_requires_bounded_action_demand_without_submission() -> None:
    text = CARD.read_text()
    assert "disables thinking" in text
    assert "2,048" in text
    assert "exploratory" in text and "16,000" in text
    assert "Do not create a competition submission" in text
    assert "V13" in text and "call 9" in text
    assert "V14" in text and "WORKER_ENVIRONMENT_KEYS" in text
