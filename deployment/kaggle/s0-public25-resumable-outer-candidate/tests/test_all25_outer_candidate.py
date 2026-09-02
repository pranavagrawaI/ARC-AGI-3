from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
RUNTIME = HERE / "all25_outer_runtime.py"
BUILDER = HERE / "build_notebook.py"
ACCEPTED_RUNTIME = (
    ROOT
    / "analysis/deployment-probes/20260831T160520Z-qwen38-r8l3b-kaggle-v14"
    / "frozen-generation-4-local/deployment/kaggle/PROTOTYPE-qwen3_8_r8_l3b_probe.py"
)
CALLBACK_TESTS = (
    HERE.parent
    / "s0-public25-resumable-candidate/tests/test_outer_callback_contract.py"
)
HARNESS = HERE.parent / "s0-public25-resumable-candidate/resumable_public25.py"


def load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def fake_callback(path: Path) -> None:
    path.write_text(
        """import argparse, os, shutil
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
a, _=p.parse_known_args()
source=os.environ.get('FAKE_CALLBACK_SOURCE')
if source:
    shutil.copy2(source, a.output)
log=Path(os.environ['FAKE_SERVER_LOG'])
count=int(os.environ.get('FAKE_SOLVER_REQUESTS', '0'))
success=os.environ.get('FAKE_REQUEST_SUCCESS', '1') == '1'
marker='POST /v1/chat/completions HTTP/1.1" 200 OK'
failure='POST /v1/chat/completions HTTP/1.1" 500 ERROR'
with log.open('a') as handle:
    for _ in range(count):
        handle.write((marker if success else failure) + '\\n')
raise SystemExit(int(os.environ.get('FAKE_CALLBACK_RC', '0')))
"""
    )


def warmup_log(runtime: ModuleType, path: Path) -> dict[str, int]:
    path.write_text(
        f"{runtime.CHAT_COMPLETION_ACCESS_MARKER}\n"
        f"{runtime.FREEZE_GC_ACCESS_MARKER}\n"
        f"{runtime.SGLANG_READY_MARKER}\n"
    )
    return runtime.server_request_facts(path.read_text())


def callback_case(
    fixtures: ModuleType,
    tmp_path: Path,
    disposition: str,
) -> tuple[Path, Path, int]:
    if disposition == "partial_failure":
        callback, checkpoint, _ = fixtures.partial_failure_fixture(tmp_path)
        return callback, checkpoint, 0

    harness = load(HARNESS, f"assembled_harness_{disposition}")
    if disposition == "partial_budget_no_game":
        predecessor = harness.RUN_ORDER[0]
        callback, checkpoint, _ = fixtures.callback_fixture(
            tmp_path,
            completed_games=[predecessor],
            attempted_this_run=[],
            completed_this_run=[],
            qualification_status="partial_budget",
        )
        return callback, checkpoint, 0
    if disposition == "partial_budget_current_game":
        game = harness.RUN_ORDER[0]
        callback, checkpoint, _ = fixtures.callback_fixture(
            tmp_path,
            completed_games=[game],
            attempted_this_run=[game],
            completed_this_run=[game],
            qualification_status="partial_budget",
        )
        return callback, checkpoint, 1
    if disposition == "complete":
        callback, checkpoint, _ = fixtures.callback_fixture(
            tmp_path,
            completed_games=list(harness.RUN_ORDER),
            attempted_this_run=list(harness.RUN_ORDER),
            completed_this_run=list(harness.RUN_ORDER),
            qualification_status="complete",
        )
        return callback, checkpoint, 1
    raise AssertionError(disposition)


def callback_args(
    tmp_path: Path,
    *,
    probe: Path,
    checkpoint: Path,
) -> SimpleNamespace:
    return SimpleNamespace(
        launcher_python=Path(sys.executable),
        resumable_public25_probe=probe,
        public25_game_probe=tmp_path / "unused-game-probe.py",
        public25_contract=tmp_path / "unused-contract.json",
        public25_template_root=tmp_path / "unused-template",
        public25_checkpoint_root=checkpoint,
        public25_workspace_root=tmp_path / "callback-workspace",
        public25_predecessor=None,
        expected_predecessor_sha256=None,
        launcher_timeout_s=30,
    )


@pytest.mark.parametrize(
    ("disposition", "qualification", "requests", "zero_label"),
    [
        (
            "partial_failure",
            "partial_failure",
            0,
            "manifest_valid_pre_request_partial_failure",
        ),
        (
            "partial_budget_no_game",
            "partial_budget",
            0,
            "manifest_valid_no_game_partial_budget",
        ),
        ("partial_budget_current_game", "partial_budget", 1, None),
        ("complete", "complete", 1, None),
    ],
)
def test_assembled_callback_dispositions_bind_checkpoint_requests_and_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    disposition: str,
    qualification: str,
    requests: int,
    zero_label: str | None,
) -> None:
    runtime = load(RUNTIME, f"assembled_runtime_{disposition}")
    fixtures = load(CALLBACK_TESTS, f"assembled_fixtures_{disposition}")
    callback, checkpoint, fixture_requests = callback_case(
        fixtures, tmp_path / "fixture", disposition
    )
    assert fixture_requests == requests
    output = tmp_path / "outer"
    output.mkdir()
    server_log = output / "server.log"
    warmup = warmup_log(runtime, server_log)
    probe = tmp_path / "fake-callback.py"
    fake_callback(probe)
    monkeypatch.setenv("FAKE_CALLBACK_SOURCE", str(callback))
    monkeypatch.setenv("FAKE_SERVER_LOG", str(server_log))
    monkeypatch.setenv("FAKE_SOLVER_REQUESTS", str(requests))

    request_result = runtime.run_resumable_public25_callback(
        callback_args(tmp_path, probe=probe, checkpoint=checkpoint),
        output_dir=output,
        base_url="http://127.0.0.1:30000",
        model_name="fixture-model",
        warmup_facts=warmup,
    )
    combined = runtime.combine_r8l3b_result(
        request_result,
        {
            "status": "passed",
            "failure_classification": None,
            "log_classification_allowed": True,
        },
    )

    assert combined["status"] == "passed"
    assert combined["qualification_status"] == qualification
    assert combined["outer_callback_contract"]["observed_solver_requests"] == requests
    assert (
        combined["outer_callback_contract"]["zero_request_disposition"]
        == zero_label
    )
    assert combined["r8_l3b_gate"] == {
        "request_path_passed": True,
        "lifecycle_passed": True,
        "teardown_log_accepted": True,
    }


def test_assembled_callback_fails_closed_on_missing_or_corrupt_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = load(RUNTIME, "assembled_runtime_negative")
    fixtures = load(CALLBACK_TESTS, "assembled_fixtures_negative")
    callback, checkpoint, _ = callback_case(
        fixtures, tmp_path / "fixture", "partial_failure"
    )
    output = tmp_path / "outer"
    output.mkdir()
    server_log = output / "server.log"
    warmup = warmup_log(runtime, server_log)
    probe = tmp_path / "fake-callback.py"
    fake_callback(probe)
    monkeypatch.setenv("FAKE_SERVER_LOG", str(server_log))

    monkeypatch.delenv("FAKE_CALLBACK_SOURCE", raising=False)
    with pytest.raises(RuntimeError, match="did not publish its result"):
        runtime.run_resumable_public25_callback(
            callback_args(tmp_path, probe=probe, checkpoint=checkpoint),
            output_dir=output,
            base_url="http://127.0.0.1:30000",
            model_name="fixture-model",
            warmup_facts=warmup,
        )

    with (checkpoint / "checkpoint.json").open("a") as handle:
        handle.write(" ")
    monkeypatch.setenv("FAKE_CALLBACK_SOURCE", str(callback))
    with pytest.raises(RuntimeError, match="checkpoint manifest hash differs"):
        runtime.run_resumable_public25_callback(
            callback_args(tmp_path, probe=probe, checkpoint=checkpoint),
            output_dir=output,
            base_url="http://127.0.0.1:30000",
            model_name="fixture-model",
            warmup_facts=warmup,
        )


def test_assembled_callback_rejects_unsuccessful_request_and_bad_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = load(RUNTIME, "assembled_runtime_request_negative")
    fixtures = load(CALLBACK_TESTS, "assembled_fixtures_request_negative")
    callback, checkpoint, _ = callback_case(
        fixtures, tmp_path / "fixture", "partial_budget_current_game"
    )
    output = tmp_path / "outer"
    output.mkdir()
    server_log = output / "server.log"
    warmup = warmup_log(runtime, server_log)
    probe = tmp_path / "fake-callback.py"
    fake_callback(probe)
    monkeypatch.setenv("FAKE_CALLBACK_SOURCE", str(callback))
    monkeypatch.setenv("FAKE_SERVER_LOG", str(server_log))
    monkeypatch.setenv("FAKE_SOLVER_REQUESTS", "1")
    monkeypatch.setenv("FAKE_REQUEST_SUCCESS", "0")
    with pytest.raises(RuntimeError, match="request-success contract differs"):
        runtime.run_resumable_public25_callback(
            callback_args(tmp_path, probe=probe, checkpoint=checkpoint),
            output_dir=output,
            base_url="http://127.0.0.1:30000",
            model_name="fixture-model",
            warmup_facts=warmup,
        )

    request_result = {"status": "passed"}
    combined = runtime.combine_r8l3b_result(
        request_result,
        {
            "status": "invalid",
            "failure_classification": "term_escalation_required",
            "log_classification_allowed": False,
        },
    )
    assert combined["status"] == "invalid"
    assert combined["r8_l3b_gate"]["request_path_passed"] is True
    assert combined["r8_l3b_gate"]["lifecycle_passed"] is False


def source_functions(path: Path) -> dict[str, str]:
    source = path.read_text()
    tree = ast.parse(source)
    result = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        value = ast.get_source_segment(source, node)
        assert value is not None
        result[node.name] = value
    return result


def test_accepted_term_only_lifecycle_functions_are_byte_identical() -> None:
    accepted = source_functions(ACCEPTED_RUNTIME)
    candidate = source_functions(RUNTIME)
    intentionally_joined = {"run_probe", "parse_args"}
    for name, source in accepted.items():
        if name not in intentionally_joined:
            assert candidate[name] == source
    accepted_source = ACCEPTED_RUNTIME.read_text()
    candidate_source = RUNTIME.read_text()
    tail_marker = "    finally:\n        shutdown: dict[str, Any] = {"
    accepted_tail = accepted_source[
        accepted_source.index(tail_marker):accepted_source.index("\n\ndef parse_args")
    ]
    candidate_tail = candidate_source[
        candidate_source.index(tail_marker):candidate_source.index("\n\ndef parse_args")
    ]
    assert candidate_tail == accepted_tail


def valid_contract(path: Path) -> None:
    harness = load(HARNESS, "assembled_builder_contract")
    value = {
        "schema": harness.CONTRACT_SCHEMA,
        "public_games": list(harness.PUBLIC_GAMES),
        "action_caps": harness.FIRST_LEVEL_ACTION_CAPS,
        "run_order": list(harness.RUN_ORDER),
        "evidence_kind": "public_progress_not_official_score",
        "admission_policy": "C1_sequential_game_shards",
        "solver_envelope": {
            "context_window": 32_768,
            "initial_actor_calls": 6,
            "initial_max_tokens": 16_000,
            "successor_actor_calls": 1,
            "successor_max_tokens": 4_096,
            "model_retries": 0,
            "mode": "single",
            "treatment": "single_decision_discipline",
        },
        "artifact_hashes": {"fixture": "1" * 64},
        "runtime_projection": {
            "initial_decision_seconds": 1,
            "successor_decision_seconds": 1,
            "per_game_overhead_seconds": 1,
            "safety_multiplier": 1,
            "source_result_sha256": "2" * 64,
        },
    }
    path.write_text(json.dumps(value) + "\n")


def checkpoint_binding(path: Path, contract: Path) -> dict[str, object]:
    harness = load(HARNESS, "assembled_binding_contract")
    contract_sha256 = harness.validate_contract(json.loads(contract.read_text()))
    value = {
        "schema": "tycho-nooa-public25-checkpoint-dataset-binding-v1",
        "dataset_id": "pranavagrawai/tycho-nooa-p25-ckpt-g1-" + "3" * 20,
        "dataset_version": 1,
        "transport_manifest_sha256": "4" * 64,
        "contract_sha256": contract_sha256,
        "predecessor_manifest_sha256": "5" * 64,
    }
    path.write_text(json.dumps(value) + "\n")
    return value


def test_builder_assembles_exact_assets_and_retains_notebook_check_true(
    tmp_path: Path,
) -> None:
    builder = load(BUILDER, "assembled_builder")
    contract = tmp_path / "public25-contract.json"
    valid_contract(contract)

    notebook = builder.build(contract_path=contract)
    builder.validate(notebook, contract_path=contract)
    cell = "".join(notebook["cells"][3]["source"])

    assert "subprocess.run(command, check=True)" in cell
    assert "--resumable-public25-probe" in cell
    assert "--public25-checkpoint-root" in cell
    assert "KAGGLE_IS_COMPETITION_RERUN" in cell
    for index, notebook_cell in enumerate(notebook["cells"]):
        if notebook_cell["cell_type"] == "code":
            compile("".join(notebook_cell["source"]), f"cell-{index}", "exec")


def test_builder_joins_exact_checkpoint_dataset_import_before_outer_launch(
    tmp_path: Path,
) -> None:
    builder = load(BUILDER, "assembled_checkpoint_builder")
    contract = tmp_path / "public25-contract.json"
    binding_path = tmp_path / "predecessor-binding.json"
    valid_contract(contract)
    binding = checkpoint_binding(binding_path, contract)

    notebook = builder.build(
        contract_path=contract,
        predecessor_binding_path=binding_path,
    )
    builder.validate(
        notebook,
        contract_path=contract,
        predecessor_binding_path=binding_path,
    )
    assets = "".join(notebook["cells"][1]["source"])
    cell = "".join(notebook["cells"][3]["source"])

    assert "/tmp/checkpoint_dataset_transport.py" in assets
    assert "/tmp/public25-predecessor-binding.json" in assets
    assert "checkpoint_dataset_transport.py', 'import'" in cell
    assert "--expected-transport-manifest-sha256" in cell
    assert binding["transport_manifest_sha256"] in cell
    assert "--expected-contract-sha256" in cell
    assert binding["contract_sha256"] in cell
    assert "--expected-predecessor-manifest-sha256" in cell
    assert binding["predecessor_manifest_sha256"] in cell
    assert "--expected-dataset-id" in cell
    assert binding["dataset_id"] in cell
    assert cell.index("checkpoint_dataset_transport.py', 'import'") < cell.index(
        "subprocess.run(command, check=True)"
    )
    assert "'--public25-predecessor', str(predecessor_root)" in cell
    assert "'--expected-predecessor-sha256'" in cell


def test_builder_preserves_fresh_generation_and_rejects_binding_drift(
    tmp_path: Path,
) -> None:
    builder = load(BUILDER, "assembled_checkpoint_builder_negative")
    contract = tmp_path / "public25-contract.json"
    binding_path = tmp_path / "predecessor-binding.json"
    valid_contract(contract)

    fresh = builder.build(contract_path=contract)
    builder.validate(fresh, contract_path=contract)
    fresh_cell = "".join(fresh["cells"][3]["source"])
    assert "checkpoint_dataset_transport.py', 'import'" not in fresh_cell
    assert "--public25-predecessor" not in fresh_cell

    binding = checkpoint_binding(binding_path, contract)
    binding["contract_sha256"] = hashlib.sha256(b"different contract").hexdigest()
    binding_path.write_text(json.dumps(binding) + "\n")
    with pytest.raises(RuntimeError, match="binding contract differs"):
        builder.build(
            contract_path=contract,
            predecessor_binding_path=binding_path,
        )
