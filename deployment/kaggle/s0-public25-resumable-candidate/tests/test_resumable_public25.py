from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[1]
MODULE_PATH = HERE / "resumable_public25.py"


def load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("resumable_public25", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def contract(module: ModuleType) -> dict[str, object]:
    return {
        "schema": module.CONTRACT_SCHEMA,
        "public_games": list(module.PUBLIC_GAMES),
        "action_caps": module.FIRST_LEVEL_ACTION_CAPS,
        "run_order": list(module.RUN_ORDER),
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
        "artifact_hashes": {"candidate": "1" * 64},
        "runtime_projection": {
            "initial_decision_seconds": 100,
            "successor_decision_seconds": 10,
            "per_game_overhead_seconds": 30,
            "safety_multiplier": 1.25,
            "source_result_sha256": "2" * 64,
        },
    }


def make_game_scratch(
    module: ModuleType, root: Path, game: str, *, actions: int, levels: int = 1
) -> dict[str, object]:
    session = root / "solver-workspaces" / f"{game}-session"
    session.mkdir(parents=True)
    trace = []
    for sequence in range(actions):
        trace.append({
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "reasoning_tokens": 3,
        })
    (session / "llm-trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in trace)
    )
    (root / "recordings").mkdir()
    (root / "recordings" / f"{game}.recording.jsonl").write_text("{}\n")
    (root / "public25-launcher.stdout.log").write_text("passed\n")
    (root / "public25-launcher.stderr.log").write_text("")
    (root / "orchestrator-command.json").write_text("{}\n")
    official = root / "ARC-AGI-3-Agents"
    official.mkdir()
    (official / "logs.log").write_text("scorecard evidence\n")
    result = {
        "status": "passed",
        "public_games": [game],
        "public_progress": {
            "game_count": 1,
            "games": {
                game: {
                    "final_state": "NOT_FINISHED",
                    "final_levels_completed": levels,
                    "max_levels_completed": levels,
                    "recorded_action_results": actions,
                }
            },
        },
        "solver_sessions": {
            "game_count": 1,
            "total_solver_actions": actions,
            "total_actor_calls": actions,
            "games": {game: {"session": str(session)}},
        },
    }
    (root / "game-result.json").write_text(json.dumps(result) + "\n")
    return result


def test_exact_roster_and_first_level_cohort_caps() -> None:
    module = load_module()
    assert len(module.PUBLIC_GAMES) == 25
    assert set(module.FIRST_LEVEL_ACTION_CAPS) == set(module.PUBLIC_GAMES)
    assert sum(module.FIRST_LEVEL_ACTION_CAPS.values()) == 1_056
    assert [module.FIRST_LEVEL_ACTION_CAPS[game] for game in module.RUN_ORDER] == sorted(
        module.FIRST_LEVEL_ACTION_CAPS.values()
    )
    assert {cap: list(module.FIRST_LEVEL_ACTION_CAPS.values()).count(cap) for cap in (24, 40, 64, 80)} == {
        24: 9,
        40: 9,
        64: 5,
        80: 2,
    }
    assert module.validate_contract(contract(module)) == module.sha256_bytes(
        module.canonical_json(contract(module))
    )


def test_contract_refuses_unmeasured_projection_and_cap_drift() -> None:
    module = load_module()
    value = contract(module)
    value["runtime_projection"]["source_result_sha256"] = "unmeasured"  # type: ignore[index]
    with pytest.raises(RuntimeError, match="not bound"):
        module.validate_contract(value)

    value = contract(module)
    value["action_caps"] = {**module.FIRST_LEVEL_ACTION_CAPS, "ls20": 80}
    with pytest.raises(RuntimeError, match="action caps differ"):
        module.validate_contract(value)


def test_completed_game_is_atomic_manifested_and_aggregated(tmp_path: Path) -> None:
    module = load_module()
    contract_hash = module.validate_contract(contract(module))
    checkpoint_root = tmp_path / "checkpoint"
    (checkpoint_root / "games").mkdir(parents=True)
    checkpoint = module.empty_checkpoint(contract_hash)
    scratch = tmp_path / "scratch"
    game = "ls20"
    actions = module.FIRST_LEVEL_ACTION_CAPS[game]
    result = make_game_scratch(module, scratch, game, actions=actions, levels=1)

    module.publish_completed_game(
        scratch, checkpoint_root, checkpoint, game=game, result=result
    )

    assert checkpoint["completed_games"] == [game]
    verified = module.verify_checkpoint(checkpoint_root, contract_hash)
    metrics = module.aggregate(verified)
    assert metrics["completed_game_count"] == 1
    assert metrics["total_solver_actions"] == actions
    assert metrics["total_actor_calls"] == actions
    assert metrics["total_prompt_tokens"] == 10 * actions
    assert metrics["total_max_public_levels_completed"] == 1
    assert metrics["public_levels_per_solver_action"] == 1 / actions
    assert game not in metrics["remaining_games"]


def test_predecessor_import_is_hash_bound_and_tamper_evident(tmp_path: Path) -> None:
    module = load_module()
    contract_hash = module.validate_contract(contract(module))
    predecessor = tmp_path / "predecessor"
    (predecessor / "games").mkdir(parents=True)
    checkpoint = module.empty_checkpoint(contract_hash)
    scratch = tmp_path / "scratch"
    game = "s5i5"
    result = make_game_scratch(
        module,
        scratch,
        game,
        actions=module.FIRST_LEVEL_ACTION_CAPS[game],
    )
    module.publish_completed_game(
        scratch, predecessor, checkpoint, game=game, result=result
    )
    manifest_hash = module.sha256_file(predecessor / "checkpoint.json")

    imported = module.import_predecessor(
        predecessor,
        tmp_path / "current",
        contract_sha256=contract_hash,
        expected_manifest_sha256=manifest_hash,
    )
    assert imported["generation"] == 1
    assert imported["predecessor_manifest_sha256"] == manifest_hash
    assert imported["completed_games"] == [game]

    bundle_file = next(
        path
        for path in (predecessor / "games" / game).rglob("*")
        if path.is_file() and path.name != "bundle.json"
    )
    bundle_file.write_bytes(bundle_file.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="inventory or bytes differ"):
        module.verify_checkpoint(predecessor, contract_hash)


def test_uncommitted_orphan_game_is_ignored_on_restart(tmp_path: Path) -> None:
    module = load_module()
    contract_hash = module.validate_contract(contract(module))
    predecessor = tmp_path / "predecessor"
    (predecessor / "games").mkdir(parents=True)
    checkpoint = module.empty_checkpoint(contract_hash)
    module.write_json_atomic(predecessor / "checkpoint.json", checkpoint)
    manifest_hash = module.sha256_file(predecessor / "checkpoint.json")

    # This is the only crash window in publication: the complete directory was
    # renamed, but checkpoint.json still names the preceding generation.
    orphan = predecessor / "games" / "ls20"
    orphan.mkdir()
    (orphan / "unbound.txt").write_text("must not be trusted\n")

    verified = module.verify_checkpoint(predecessor, contract_hash)
    assert verified["completed_games"] == []
    imported = module.import_predecessor(
        predecessor,
        tmp_path / "current",
        contract_sha256=contract_hash,
        expected_manifest_sha256=manifest_hash,
    )
    assert imported["completed_games"] == []
    assert list((tmp_path / "current" / "games").iterdir()) == []


def test_nonwinning_game_must_reach_its_exact_cap(tmp_path: Path) -> None:
    module = load_module()
    game = "sb26"
    result = make_game_scratch(module, tmp_path, game, actions=1, levels=0)
    with pytest.raises(RuntimeError, match="stopped before its cap"):
        module.result_metrics(result, game, module.FIRST_LEVEL_ACTION_CAPS[game])


def test_projection_reserves_measured_whole_game_time() -> None:
    module = load_module()
    value = contract(module)
    # ceil((100 + 23*10 + 30) * 1.25)
    assert module.projected_game_seconds(value, "ls20") == 450


def test_fake_all25_run_publishes_exact_complete_checkpoint(tmp_path: Path) -> None:
    module = load_module()
    value = contract(module)
    value["runtime_projection"] = {
        "initial_decision_seconds": 0.01,
        "successor_decision_seconds": 0.01,
        "per_game_overhead_seconds": 0.01,
        "safety_multiplier": 1.01,
        "source_result_sha256": "3" * 64,
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(value) + "\n")
    template = tmp_path / "template"
    official = template / "ARC-AGI-3-Agents"
    official.mkdir(parents=True)
    (official / ".env").write_text("RECORDINGS_DIR=/old/path\n")
    (template / "run_official_agent.py").write_text("# fixture\n")
    fake_probe = tmp_path / "fake_probe.py"
    fake_probe.write_text(
        """import argparse, json, os
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--base-url'); p.add_argument('--model')
p.add_argument('--workspace-root', type=Path); p.add_argument('--output', type=Path)
p.add_argument('--timeout-s')
a=p.parse_args(); game=os.environ['TYCHO_PUBLIC_GAME_IDS']; cap=int(os.environ['TYCHO_KAGGLE_MAX_ACTIONS'])
session=a.workspace_root/f'{game}-session'; session.mkdir(parents=True)
(session/'llm-trace.jsonl').write_text(''.join(json.dumps({'prompt_tokens':10,'completion_tokens':4,'reasoning_tokens':3})+'\\n' for _ in range(cap)))
scratch=a.output.parent; (scratch/'recordings').mkdir(exist_ok=True)
(scratch/'recordings'/f'{game}.recording.jsonl').write_text('{}\\n')
(scratch/'public25-launcher.stdout.log').write_text('passed\\n')
(scratch/'public25-launcher.stderr.log').write_text('')
(scratch/'ARC-AGI-3-Agents'/'logs.log').write_text('scorecard\\n')
result={'status':'passed','public_games':[game],
'public_progress':{'game_count':1,'games':{game:{'final_state':'NOT_FINISHED','final_levels_completed':1,'max_levels_completed':1,'recorded_action_results':cap}}},
'solver_sessions':{'game_count':1,'total_solver_actions':cap,'total_actor_calls':cap,'games':{game:{'session':str(session)}}}}
a.output.write_text(json.dumps(result)+'\\n')
"""
    )
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "result.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "--base-url", "http://127.0.0.1:30000/v1",
            "--model", "fake",
            "--workspace-root", str(tmp_path / "work"),
            "--output", str(output),
            "--timeout-s", "2000",
            "--contract", str(contract_path),
            "--template-root", str(template),
            "--game-probe", str(fake_probe),
            "--checkpoint-root", str(checkpoint),
            "--stop-reserve-s", "0",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text())
    assert result["status"] == "passed"
    assert result["qualification_status"] == "complete"
    assert result["all25_complete"] is True
    assert result["games_attempted_this_run"] == list(module.RUN_ORDER)
    assert result["games_completed_this_run"] == list(module.RUN_ORDER)
    assert result["metrics"]["completed_game_count"] == 25
    assert result["metrics"]["total_solver_actions"] == 1_056
    assert result["metrics"]["total_max_public_levels_completed"] == 25
    assert result["metrics"]["official_score"] is None
    verified = module.verify_checkpoint(
        checkpoint, module.validate_contract(value)
    )
    assert verified["completed_games"] == list(module.PUBLIC_GAMES)


def test_run_game_timeout_contains_the_probe_process_group(tmp_path: Path) -> None:
    module = load_module()
    template = tmp_path / "template"
    official = template / "ARC-AGI-3-Agents"
    official.mkdir(parents=True)
    (official / ".env").write_text("RECORDINGS_DIR=/old/path\n")
    (template / "run_official_agent.py").write_text("# fixture\n")
    fake_probe = tmp_path / "hanging_probe.py"
    fake_probe.write_text(
        """import argparse, subprocess, sys, time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--base-url'); p.add_argument('--model')
p.add_argument('--workspace-root'); p.add_argument('--output', type=Path)
p.add_argument('--timeout-s'); a=p.parse_args()
pid_path=a.output.parent/'grandchild.pid'
code=("import os,time; from pathlib import Path; "
      f"Path({str(pid_path)!r}).write_text(str(os.getpid())); time.sleep(60)")
subprocess.Popen([sys.executable, '-c', code])
print('probe-started-grandchild', flush=True)
time.sleep(60)
"""
    )
    scratch = tmp_path / "scratch"
    args = SimpleNamespace(
        template_root=template,
        base_url="http://127.0.0.1:30000/v1",
        model="no-model-request",
        game_probe=fake_probe,
    )

    completed, result = module.run_game(
        game="bp35",
        timeout_s=0.1,
        args=args,
        scratch=scratch,
        communication_grace_s=0,
        termination_grace_s=0.5,
    )

    assert completed.returncode == 124
    assert result is None
    command = json.loads((scratch / "orchestrator-command.json").read_text())
    assert command["timed_out"] is True
    assert command["process_group_closed"] is True
    assert command["term_sent"] is True
    assert "probe-started-grandchild" in command["stdout"]
    grandchild_pid = int((scratch / "grandchild.pid").read_text())
    deadline = time.monotonic() + 2
    while Path(f"/proc/{grandchild_pid}").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not Path(f"/proc/{grandchild_pid}").exists()
