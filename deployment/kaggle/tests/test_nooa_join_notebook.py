from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

CANDIDATE = Path(__file__).resolve().parents[1] / "s0-nooa-join-candidate"
BUILDER = CANDIDATE / "build_notebook.py"
NOTEBOOK = CANDIDATE / "s0_nooa_join_public3.ipynb"
METADATA = CANDIDATE / "kernel-metadata.json"


def load_builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location("nooa_join_builder", BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_notebook_is_deterministic_compiles_and_refuses_submission() -> None:
    builder = load_builder()
    first = builder.build()
    second = builder.build()
    assert first == second
    builder.validate(first)
    NOTEBOOK.write_text(json.dumps(first, indent=1) + "\n")
    assert first == json.loads(NOTEBOOK.read_text())
    source = "\n".join("".join(cell.get("source", [])) for cell in first["cells"])
    metadata = json.loads(METADATA.read_text())
    assert metadata["is_private"] is True
    assert metadata["enable_internet"] is False
    assert metadata["id"] == "pranavagrawai/prototype-nooa-join-public-3"
    assert "ls20,s5i5,cd82" in source
    assert "TYCHO_KAGGLE_MAX_ACTIONS'] = '16'" in source
    assert "'--launcher-timeout-s', '7200'" in source
    assert "NOOA join qualifier refuses competition rerun mode" in source
    assert "kaggle competitions submit" not in source
    assert "OPERATION_MODE=offline" in source
    assert "OPERATION_MODE=competition" not in source
    assert "nooa-join.zip" in source
    assert "nooa_public_probe.py" in source


def test_builder_embeds_parameterized_games_and_caps() -> None:
    builder = load_builder()
    games = (
        "ar25,bp35,cd82,cn04,dc22,ft09,g50t,ka59,lf52,lp85,"
        "ls20,m0r0,r11l,re86,s5i5,sb26,sc25,sk48,sp80,su15,"
        "tn36,tr87,tu93,vc33,wa30"
    )
    notebook = builder.build(
        games=games,
        max_actions="8",
        launcher_timeout_s=21600,
        worker_timeout_s=1800,
        heading="# CONTACT\n",
    )
    builder.validate(
        notebook, games=games, max_actions="8", launcher_timeout_s=21600
    )
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "TYCHO_PUBLIC_GAME_IDS'] = " in source
    assert "TYCHO_KAGGLE_MAX_ACTIONS'] = '8'" in source
    assert "'--launcher-timeout-s', '21600'" in source
    assert "TYCHO_NOOA_WORKER_TIMEOUT_SECONDS'] = '1800'" in source
    assert source.startswith("# CONTACT") or "# CONTACT" in source
    assert "kaggle competitions submit" not in source
