#!/usr/bin/env python3
"""Matched live Builder replay for the theory-transaction gate."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from nooa.unifiedllm import UnifiedLLM
from tycho_nooa import TychoNooaAgent
from tycho_nooa.builder import BuilderConfig
from tycho_nooa.llm import build_llm
from tycho_nooa.theory_evaluation import EvidenceFrontier, SandboxTheoryEvaluator
from tycho_nooa.wm_probe import ReadOnlyWorkspacePythonSandbox

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
SHARED_PATH = ROOT / "analysis/builder-continuation/20260822-matched-budget-awareness/run.py"
SOURCE = ROOT / "runs/tycho-nooa-trigger-replacement-gemma4-31b-ar25-20260821T154601Z/game"
SOURCE_CALL = "000491"
START_SHA = "4f78ef3b371d3e8f4d106710d19d03f94de6c60cfa399776818714fa1c5a86df"
MODEL = "openai/deepseek/deepseek-v4-flash-0731"
ARMS = ("builder_transactional",)
TREATMENTS = {"builder_control": "control", "builder_transactional": "transactional_builder"}
MAX_TOOL_STEPS = 12
MAX_TOKENS = 16384
MAX_CALLS_PER_ARM = 14
MAX_CALLS_TOTAL = 14
MAX_SECONDS_PER_ARM = 1800.0
MAX_COST_USD = 0.10
RESERVE_PER_CALL_USD = 0.01
EXPECTED_PROVIDER = "Sail Research"


def _load_shared():
    spec = importlib.util.spec_from_file_location("budget_replay_shared", SHARED_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load provider-budget support")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.MAX_CALLS_PER_ARM = MAX_CALLS_PER_ARM
    module.MAX_CALLS_TOTAL = MAX_CALLS_TOTAL
    module.MAX_COST_USD = MAX_COST_USD
    module.RESERVE_PER_CALL_USD = RESERVE_PER_CALL_USD
    module.PROVIDER_ENDPOINT = "sail-research/fp4"
    module.EXPECTED_PROVIDER = EXPECTED_PROVIDER
    return module


shared = _load_shared()


async def main(preflight_only: bool) -> None:
    _validate_environment()
    live, kind = build_llm()
    if kind != "live" or live.model != MODEL:
        raise RuntimeError(f"provider/model drift: {kind} {live.model}")
    source_manifest = shared._read_json(shared._call_path(SOURCE_CALL) / "workspace.json")
    try:
        if preflight_only:
            with tempfile.TemporaryDirectory(prefix="theory-tx-provider-") as temp:
                rows = [_prepare(Path(temp) / arm, arm, source_manifest) for arm in ARMS]
                manifest = _manifest(rows, preflight=True)
                _validate_parity(manifest)
                print(json.dumps({"status": "preflight_pass", **manifest}, indent=2))
            return

        if (OUTPUT / "arms").exists() or (OUTPUT / "result.json").exists():
            raise FileExistsError("refusing to overwrite experiment evidence")
        (OUTPUT / "arms").mkdir()
        preparations = [_prepare(OUTPUT / "arms" / arm, arm, source_manifest) for arm in ARMS]
        manifest = _manifest(preparations, preflight=False)
        _validate_parity(manifest)
        _write_json(OUTPUT / "manifest.json", manifest)
        budget = shared.BudgetState(OUTPUT / "budget.json")
        budget.persist()
        provider_trace = OUTPUT / "provider-calls.jsonl"
        results = []
        for arm in ARMS:
            results.append(await _run_arm(OUTPUT / "arms" / arm, arm, live, budget, provider_trace))
        result = _assess(manifest, results, budget, provider_trace)
        _write_json(OUTPUT / "result.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        if result["status"] != "complete" or result["gate"] != "pass":
            raise SystemExit(2)
    finally:
        await live.aclose()


def _prepare(root: Path, arm: str, source_manifest: dict[str, Any]) -> dict[str, Any]:
    workspace_root = root / "workspace"
    workspace = workspace_root / "ar25"
    shared._materialize(source_manifest, workspace)
    (workspace / "world_model.py").write_bytes(shared._blob(START_SHA).read_bytes())
    if _sha(workspace / "world_model.py") != START_SHA:
        raise RuntimeError("starting-model commitment drift")
    frontier = EvidenceFrontier.freeze(workspace)
    evaluation = _evaluator().evaluate((workspace / "world_model.py").read_bytes(), frontier)
    if len(_exact_ids(evaluation)) != 10:
        raise RuntimeError(f"expected 10 typed exact transitions, got {len(_exact_ids(evaluation))}")
    return {
        "arm": arm,
        "treatment": TREATMENTS[arm],
        "workspace_root": str(workspace_root),
        "workspace_commitment": shared._tree_commitment(workspace),
        "evidence_frontier_sha256": frontier.sha256,
        "starting_model_sha256": START_SHA,
        "starting_evaluation": _jsonable(evaluation),
    }


async def _run_arm(root: Path, arm: str, delegate: UnifiedLLM, budget: Any, provider_trace: Path) -> dict[str, Any]:
    workspace_root = root / "workspace"
    workspace = workspace_root / "ar25"
    capped = shared.BudgetedLLM(delegate, arm=arm, budget=budget, trace_path=provider_trace)
    turn = shared._turn18()
    before_files = _artifact_hashes(workspace)
    before_frontier = EvidenceFrontier.freeze(workspace)
    before_eval = _evaluator().evaluate((workspace / "world_model.py").read_bytes(), before_frontier)
    status, error, exposure = "complete", None, None
    started = time.monotonic()
    with _arm_environment(TREATMENTS[arm]):
        agent = TychoNooaAgent(
            llm=capped,
            game_id="ar25",
            workspace_root=workspace_root,
            available_actions=turn.available_actions,
            tool_trace_path=root / "actor-tool-trace.jsonl",
            resume_workspace=True,
            builder_config=BuilderConfig(
                max_tool_steps=MAX_TOOL_STEPS,
                max_tokens=MAX_TOKENS,
                show_budget_telemetry=True,
                commit_seconds_remaining=180,
            ),
        )
        trigger = agent._trigger_builder
        if trigger is None:
            raise RuntimeError("trigger Builder unavailable")
        try:
            async with asyncio.timeout(MAX_SECONDS_PER_ARM + 5):
                exposure = await trigger._invoke_builder(
                    turn,
                    "Repair the first wrong graded transition at this preserved 8/16 frontier without losing any exact transition.",
                    deadline_at=started + MAX_SECONDS_PER_ARM,
                )
        except TimeoutError as caught:
            status, error = "timeout", f"{type(caught).__name__}: {caught}"
        except shared.BudgetStop as caught:
            status, error = "budget_stop", str(caught)
        except Exception as caught:  # preserve all failure evidence
            status, error = "error", f"{type(caught).__name__}: {caught}"
    after_frontier = EvidenceFrontier.freeze(workspace)
    if after_frontier.sha256 != before_frontier.sha256:
        status, error = "frontier_drift", "Builder changed frozen public evidence"
    after_eval = _evaluator().evaluate((workspace / "world_model.py").read_bytes(), before_frontier)
    ledger_path = workspace.parent / ".theory-transactions" / workspace.name / "ledger.jsonl"
    ledger = _jsonl(ledger_path)
    calls = [row for row in _jsonl(provider_trace) if row.get("arm") == arm]
    result = {
        "arm": arm,
        "treatment": TREATMENTS[arm],
        "status": status,
        "error": error,
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "provider_calls": len(calls),
        "observed_cost_usd": round(sum(float(row["usage"]["cost"]) for row in calls if isinstance(row.get("usage"), dict) and isinstance(row["usage"].get("cost"), (int, float))), 12),
        "starting_model_sha256": START_SHA,
        "final_model_sha256": _sha(workspace / "world_model.py"),
        "before_evaluation": _jsonable(before_eval),
        "after_evaluation": _jsonable(after_eval),
        "lost_exact": sorted(_exact_ids(before_eval) - _exact_ids(after_eval)),
        "gained_exact": sorted(_exact_ids(after_eval) - _exact_ids(before_eval)),
        "artifact_hashes_before": before_files,
        "artifact_hashes_after": _artifact_hashes(workspace),
        "ledger_events": [row.get("event") for row in ledger],
        "ledger": ledger,
        "publication_decision": _jsonable(exposure.publication_decision) if exposure is not None else None,
        "publication_feedback": exposure.publication_feedback if exposure is not None else None,
        "builder_result": _jsonable(exposure.result) if exposure is not None else None,
    }
    _write_json(root / "result.json", result)
    return result


def _assess(manifest: dict[str, Any], arms: list[dict[str, Any]], budget: Any, provider_trace: Path) -> dict[str, Any]:
    by_arm = {row["arm"]: row for row in arms}
    provider_rows = [row for row in _jsonl(provider_trace) if row.get("status") == "complete"]
    routes = {json.dumps(row.get("provider_route", {}), sort_keys=True) for row in provider_rows}
    invalid = []
    if set(by_arm) != set(ARMS): invalid.append("arm missing")
    if any(row["status"] != "complete" for row in arms): invalid.append("arm did not complete")
    if budget.stop_reason is not None: invalid.append(f"budget/provider stop: {budget.stop_reason}")
    if len(routes) != 1: invalid.append("provider route unavailable or inconsistent")
    if provider_rows and any(row.get("provider_route", {}).get("reported_provider") != EXPECTED_PROVIDER for row in provider_rows): invalid.append("unexpected provider")
    tx = by_arm.get("builder_transactional", {})
    if "candidate_assessed" not in tx.get("ledger_events", []): invalid.append("transaction assessed no semantic edit")
    tx_lost = tx.get("lost_exact", [])
    gate = "pass" if not invalid and not tx_lost else "fail"
    if tx_lost: invalid.append("transaction persisted protected exact-transition loss")
    return {
        "schema": "tycho-nooa-theory-transaction-provider-replay-v1",
        "status": "complete" if not invalid else "invalid",
        "gate": gate,
        "invalid_reasons": invalid,
        "manifest_commitment": shared._commitment(manifest),
        "arms": arms,
        "budget": shared._read_json(budget.path),
        "observed_provider_routes": [json.loads(route) for route in sorted(routes)],
        "claim_ceiling": "one live transactional Builder regression on preserved AR25 evidence; zero ARC actions, no gameplay or transfer claim",
    }


def _manifest(rows: list[dict[str, Any]], preflight: bool) -> dict[str, Any]:
    return {
        "schema": "tycho-nooa-theory-transaction-provider-replay-manifest-v1",
        "preflight": preflight,
        "source_run": str(SOURCE.parent.relative_to(ROOT)),
        "source_call": SOURCE_CALL,
        "source_commit": _git_commit(),
        "experiment_card_sha256": _sha(OUTPUT / "EXPERIMENT-CARD.md"),
        "support_script": str(SHARED_PATH.relative_to(ROOT)),
        "support_script_sha256": _sha(SHARED_PATH),
        "model": MODEL,
        "provider": {"endpoint": shared.PROVIDER_ENDPOINT, "allow_fallbacks": False, "expected": EXPECTED_PROVIDER},
        "order": list(ARMS),
        "builder": {"max_tool_steps": MAX_TOOL_STEPS, "max_tokens": MAX_TOKENS, "max_seconds_per_arm": MAX_SECONDS_PER_ARM, "commit_seconds_remaining": 180},
        "budget": {"max_calls_per_arm": MAX_CALLS_PER_ARM, "max_calls_total": MAX_CALLS_TOTAL, "max_cost_usd": MAX_COST_USD, "reserve_per_call_usd": RESERVE_PER_CALL_USD},
        "execution": {"arc_actions": 0},
        "arms": rows,
    }


def _validate_parity(manifest: dict[str, Any]) -> None:
    rows = manifest["arms"]
    for field in ("workspace_commitment", "evidence_frontier_sha256", "starting_model_sha256", "starting_evaluation"):
        if len({json.dumps(row[field], sort_keys=True) for row in rows}) != 1:
            raise RuntimeError(f"arm parity drift: {field}")


def _validate_environment() -> None:
    expected = {"ARC_LLM_MODEL": "deepseek/deepseek-v4-flash-0731", "ARC_LLM_CONTEXT_WINDOW": "1048576", "ARC_LLM_MAX_TOKENS": "16384", "ARC_LLM_USE_RESPONSES": "0", "TYCHO_NOOA_REQUEST_REASONING": "1", "TYCHO_SANDBOX_RUNTIME": "docker", "ARC_LLM_PROVIDER_ENDPOINT": "sail-research/fp4"}
    drift = {key: os.environ.get(key) for key, value in expected.items() if os.environ.get(key) != value}
    if drift: raise RuntimeError(f"experiment environment drift: {drift}")
    if not os.environ.get("ARC_LLM_BASE_URL") or not os.environ.get("ARC_LLM_API_KEY"): raise RuntimeError("gateway configuration absent")
    if len((OUTPUT / "EXPERIMENT-CARD.md").read_bytes()) > 1024: raise RuntimeError("experiment card exceeds 1 KiB")


@contextmanager
def _arm_environment(treatment: str):
    names = ("TYCHO_MODE", "TYCHO_NOOA_TREATMENT")
    prior = {name: os.environ.get(name) for name in names}
    os.environ["TYCHO_MODE"] = "trigger"
    os.environ["TYCHO_NOOA_TREATMENT"] = treatment
    try: yield
    finally:
        for name, value in prior.items():
            if value is None: os.environ.pop(name, None)
            else: os.environ[name] = value


def _evaluator() -> SandboxTheoryEvaluator:
    return SandboxTheoryEvaluator(ReadOnlyWorkspacePythonSandbox())


def _exact_ids(evaluation: Any) -> set[str]:
    return {
        item.obligation_id
        for item in evaluation.obligations
        if item.kind.value == "transition" and item.status.value == "passed"
    }


def _artifact_hashes(workspace: Path) -> dict[str, str | None]:
    return {name: _sha(workspace / name) for name in ("world_model.py", "notes/world_model.md", "notes/world_model_report.md", "plan.py")}


def _jsonable(value: Any) -> Any:
    if is_dataclass(value): return _jsonable(asdict(value))
    if hasattr(value, "model_dump"): return _jsonable(value.model_dump(mode="json"))
    if isinstance(value, dict): return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)): return sorted((_jsonable(v) for v in value), key=lambda x: json.dumps(x, sort_keys=True))
    if isinstance(value, (list, tuple)): return [_jsonable(v) for v in value]
    if hasattr(value, "value"): return _jsonable(value.value)
    if value is None or isinstance(value, (str, int, float, bool)): return value
    return str(value)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file(): return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _git_commit() -> str:
    import subprocess
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.preflight_only))
