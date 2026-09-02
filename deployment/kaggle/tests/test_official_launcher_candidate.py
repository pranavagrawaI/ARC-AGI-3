from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from types import ModuleType

DEPLOYMENT_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_ROOT = DEPLOYMENT_ROOT / "s0-official-launcher-kernel"
BUILDER_PATH = CANDIDATE_ROOT / "build_notebook.py"
NOTEBOOK_PATH = CANDIDATE_ROOT / "s0_official_launcher.ipynb"
PROBE_PATH = CANDIDATE_ROOT / "official_launcher_probe.py"


def load_builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location("official_launcher_builder", BUILDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_candidate_is_deterministic_and_compiles() -> None:
    builder = load_builder()
    first = builder.build()
    second = builder.build()

    assert first == second
    builder.validate(first)
    assert first == json.loads(NOTEBOOK_PATH.read_text())
    for index, cell in enumerate(first["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"candidate-cell-{index}", "exec")


def test_candidate_binds_the_qualified_inputs_and_competition_mode() -> None:
    builder = load_builder()
    notebook = builder.build()
    metadata = json.loads((CANDIDATE_ROOT / "kernel-metadata.json").read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert metadata["is_private"] is True
    assert metadata["enable_internet"] is False
    assert metadata["machine_shape"] == "NvidiaRtxPro6000"
    assert (
        "pranavagrawai/tycho-nooa-solver-wheelhouse-cp312-v2"
        in metadata["dataset_sources"]
    )
    assert builder.WHEELHOUSE_MANIFEST_SHA256 in source
    assert "'dataset_version': 2" in source
    assert "OPERATION_MODE=competition" in source
    assert "OPERATION_MODE=online" not in source
    assert "run_official_agent.py" in source
    assert "subprocess.run(command, check=True)" in source
    assert "KAGGLE_IS_COMPETITION_RERUN" in source


def test_downloaded_official_source_matches_builder_pins() -> None:
    builder = load_builder()
    source_root = (
        builder.ROOT
        / "analysis/deployment-probes/20260901T052441Z-official-launcher-candidate-local"
        / "official-source"
    )
    flattened = {
        ".env.example": ".env.example",
        ".git/logs/HEAD": "HEAD",
        "agents/agent.py": "agent.py",
        "agents/recorder.py": "recorder.py",
        "agents/swarm.py": "swarm.py",
        "agents/templates/random_agent.py": "random_agent.py",
        "agents/tracing.py": "tracing.py",
        "main.py": "main.py",
    }
    for mounted_name, downloaded_name in flattened.items():
        actual = hashlib.sha256((source_root / downloaded_name).read_bytes()).hexdigest()
        assert actual == builder.OFFICIAL_SOURCE_HASHES[mounted_name]
    assert builder.OFFICIAL_REVISION in (source_root / "HEAD").read_text()


def test_launcher_probe_refuses_noncompetition_execution(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(PROBE_PATH),
            "--base-url",
            "http://127.0.0.1:30000/v1",
            "--model",
            "model",
            "--workspace-root",
            str(tmp_path / "workspace"),
            "--output",
            str(tmp_path / "result.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
        env={},
    )

    assert completed.returncode != 0
    assert "competition rerun" in completed.stderr
    assert not (tmp_path / "result.json").exists()


def test_exact_official_main_and_swarm_complete_without_arc_api(tmp_path: Path) -> None:
    builder = load_builder()
    source_root = (
        builder.ROOT
        / "analysis/deployment-probes/20260901T052441Z-official-launcher-candidate-local"
        / "official-source"
    )
    official_root = tmp_path / "ARC-AGI-3-Agents"
    templates = official_root / "agents/templates"
    templates.mkdir(parents=True)
    for source_name, relative in (
        ("main.py", "main.py"),
        ("agent.py", "agents/agent.py"),
        ("recorder.py", "agents/recorder.py"),
        ("swarm.py", "agents/swarm.py"),
        ("tracing.py", "agents/tracing.py"),
    ):
        (official_root / relative).write_bytes((source_root / source_name).read_bytes())
    (templates / "my_agent.py").write_bytes(builder.ADAPTER.read_bytes())
    (official_root / "agents/__init__.py").write_text(
        "from .agent import Agent, Playback\n"
        "from .swarm import Swarm\n"
        "from .templates.my_agent import MyAgent\n"
        "AVAILABLE_AGENTS = {'myagent': MyAgent}\n"
    )
    (official_root / ".env.example").write_text("")
    (official_root / ".env").write_text(
        "SCHEME=http\nHOST=gateway\nPORT=8001\n"
        "ARC_API_KEY=test-key-123\nARC_BASE_URL=http://gateway:8001/\n"
        "OPERATION_MODE=competition\n"
    )
    (official_root / "local_solver.py").write_text(
        textwrap.dedent(
            """
            from tycho_nooa.models import ActionDecision

            class FakeSolver:
                def decide(self, turn):
                    return ActionDecision(action="UP")
                def observe(self, result):
                    pass
                def finish(self, final_observation, actions):
                    pass
                def close(self):
                    pass

            def create_solver(**kwargs):
                return FakeSolver()
            """
        )
    )

    stubs = tmp_path / "stubs"
    (stubs / "arc_agi").mkdir(parents=True)
    (stubs / "arc_agi/scorecard.py").write_text(
        "class EnvironmentScorecard:\n"
        "    def model_dump(self): return {'status': 'closed'}\n"
        "    def get(self, game_id): return {'game_id': game_id}\n"
    )
    (stubs / "arc_agi/__init__.py").write_text(
        textwrap.dedent(
            """
            from enum import Enum
            import numpy as np
            from arcengine import FrameDataRaw, GameAction, GameState
            from .scorecard import EnvironmentScorecard

            class OperationMode(Enum):
                ONLINE = "online"
                COMPETITION = "competition"

            class EnvironmentWrapper:
                pass

            def frame(state, value=None, levels=0):
                out = FrameDataRaw(
                    game_id="fake-game", state=state, levels_completed=levels,
                    win_levels=1, available_actions=[GameAction.ACTION1.value],
                )
                if value is not None:
                    out.frame = [np.array([[value]], dtype=np.int8)]
                return out

            class FakeEnvironment:
                def __init__(self):
                    self.observation_space = frame(GameState.NOT_PLAYED)
                def step(self, action, *, data, reasoning):
                    if action is GameAction.RESET:
                        self.observation_space = frame(GameState.NOT_FINISHED, 1)
                    else:
                        self.observation_space = frame(GameState.WIN, 2, 1)
                    return self.observation_space

            class Arcade:
                operation_mode = OperationMode.COMPETITION
                def make(self, game_id, scorecard_id=None):
                    return FakeEnvironment()
                def open_scorecard(self, tags=None):
                    return "fake-card"
                def close_scorecard(self, card_id):
                    return EnvironmentScorecard()
            """
        )
    )
    (stubs / "requests.py").write_text(
        textwrap.dedent(
            """
            class Response:
                status_code = 200
                text = ""
                def json(self): return [{"game_id": "fake-game"}]
            class Session:
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def __init__(self): self.headers = Header()
                def get(self, *args, **kwargs): return Response()
            class Header(dict):
                pass
            class exceptions:
                class RequestException(Exception): pass
            """
        )
    )

    runner = DEPLOYMENT_ROOT / "runtime/run_official_agent.py"
    python_path = ":".join((str(stubs), str(official_root), str(DEPLOYMENT_ROOT)))
    completed = subprocess.run(
        [sys.executable, str(runner), str(official_root / "main.py"), "--agent", "myagent"],
        cwd=official_root,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": python_path},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Game list: ['fake-game']" in completed.stdout
    assert "***** MAKING ALL AGENTS" in completed.stdout
