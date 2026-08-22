#!/usr/bin/env python3
"""Run the capped turn-18 canonical-vs-retained Builder continuation."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nooa.unifiedllm import FakeLLMClient, LLMResponse, Tool, UnifiedLLM
from pydantic import BaseModel
from tycho_nooa import GameTurn, TychoNooaAgent
from tycho_nooa.builder import (
    BuilderConfig,
    BuilderRequest,
    FocusedBuilder,
    verify_state,
)
from tycho_nooa.decision_checkpoints import _replay_context
from tycho_nooa.llm import build_llm
from tycho_nooa.llm_trace import LLMTrace
from tycho_nooa.shadow_eval import _provider_route

ROOT = Path(__file__).resolve().parents[3]
SOURCE = (
    ROOT
    / "runs/tycho-nooa-trigger-replacement-gemma4-31b-ar25-20260821T154601Z"
    / "game"
)
OUTPUT = Path(__file__).resolve().parent
SOURCE_CALL = "000491"
CANDIDATE_CALL = "000492"
CANONICAL_SHA = "060648964ad3943f0083650a8b5fa382986a18a7c59e440db17ba1d959788905"
CANDIDATE_SHA = "4f78ef3b371d3e8f4d106710d19d03f94de6c60cfa399776818714fa1c5a86df"
MODEL = "openai/google/gemma-4-31b-it"
ORDER = ("retained_candidate", "canonical")
STARTING_MODELS = {"canonical": CANONICAL_SHA, "retained_candidate": CANDIDATE_SHA}
MAX_TOOL_STEPS = 12
MAX_TOKENS = 4096
MAX_CALLS_PER_ARM = 14
MAX_CALLS_TOTAL = 28
MAX_SECONDS_PER_ARM = 240.0
MAX_COST_USD = 0.42
RESERVE_PER_CALL_USD = 0.015
PROVIDER_ENDPOINT = "coreweave/bf16"
EXPECTED_PROVIDER = "CoreWeave"
ACCURACY = re.compile(
    r"^\[verify state\] simulation_accuracy=([0-9.]+);",
    re.MULTILINE,
)
REPLAY = re.compile(
    r"^\[verify state\] .*?matched=(\d+)/evaluated=(\d+)/observed=(\d+)",
    re.MULTILINE,
)
ACCURACY_AUTO = re.compile(
    r"^  (?:accepted|exact)_transition_accuracy \(simulation_accuracy\) = ([0-9.]+)",
    re.MULTILINE,
)
REPLAY_AUTO = re.compile(
    r"^  transition_replay = matched (\d+) / evaluated (\d+) / observed (\d+)",
    re.MULTILINE,
)
FIRST_DIVERGENCE = re.compile(r"^  first divergence: (.+)$", re.MULTILINE)


class BudgetStop(RuntimeError):
    """A precommitted provider-call, spend, or evidence stop fired."""


@dataclass
class BudgetState:
    path: Path
    calls: int = 0
    observed_cost_usd: float = 0.0
    stop_reason: str | None = None
    calls_by_arm: dict[str, int] = field(default_factory=dict)

    def reserve(self, arm: str) -> tuple[int, int]:
        arm_calls = self.calls_by_arm.get(arm, 0)
        if arm_calls >= MAX_CALLS_PER_ARM:
            self.stop_reason = f"{arm}:arm_call_cap"
            self.persist()
            raise BudgetStop(self.stop_reason)
        if self.calls >= MAX_CALLS_TOTAL:
            self.stop_reason = "total_call_cap"
            self.persist()
            raise BudgetStop(self.stop_reason)
        if self.observed_cost_usd + RESERVE_PER_CALL_USD > MAX_COST_USD:
            self.stop_reason = "insufficient_reserved_cost"
            self.persist()
            raise BudgetStop(self.stop_reason)
        self.calls += 1
        arm_calls += 1
        self.calls_by_arm[arm] = arm_calls
        self.persist()
        return self.calls, arm_calls

    def observe_cost(self, cost: float | None) -> None:
        if cost is None:
            self.stop_reason = "missing_observed_cost"
            self.persist()
            raise BudgetStop(self.stop_reason)
        self.observed_cost_usd += cost
        if self.observed_cost_usd > MAX_COST_USD:
            self.stop_reason = "observed_cost_exceeded"
            self.persist()
            raise BudgetStop(self.stop_reason)
        self.persist()

    def persist(self) -> None:
        _write_json(
            self.path,
            {
                "max_calls_total": MAX_CALLS_TOTAL,
                "max_calls_per_arm": MAX_CALLS_PER_ARM,
                "max_cost_usd": MAX_COST_USD,
                "reserve_per_call_usd": RESERVE_PER_CALL_USD,
                "calls": self.calls,
                "calls_by_arm": self.calls_by_arm,
                "observed_cost_usd": round(self.observed_cost_usd, 12),
                "remaining_cost_usd": round(
                    max(0.0, MAX_COST_USD - self.observed_cost_usd), 12
                ),
                "stop_reason": self.stop_reason,
            },
        )


class BudgetedLLM(UnifiedLLM):
    """Delegate to the live client only after reserving global and arm budget."""

    def __init__(
        self,
        delegate: UnifiedLLM,
        *,
        arm: str,
        budget: BudgetState,
        trace_path: Path,
    ) -> None:
        super().__init__(delegate.model, **delegate.config)
        self._delegate = delegate
        self._arm = arm
        self._budget = budget
        self._trace_path = trace_path
        self.cache_control_injection_points = delegate.cache_control_injection_points

    def call(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        output_model: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        del messages, tools, output_model, kwargs
        raise RuntimeError("matched continuation uses async provider calls only")

    async def acall(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        output_model: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        global_index, arm_index = self._budget.reserve(self._arm)
        if "extra_body" in kwargs:
            raise RuntimeError("provider routing override collision")
        kwargs["extra_body"] = {
            "provider": {
                "order": [PROVIDER_ENDPOINT],
                "allow_fallbacks": False,
            }
        }
        started = time.monotonic()
        request = {
            "messages": messages,
            "tools": _tool_schemas(tools or []),
            "output_model": getattr(output_model, "__name__", None),
            "parameters": kwargs,
        }
        base = {
            "global_call_index": global_index,
            "arm_call_index": arm_index,
            "arm": self._arm,
            "request_commitment": _commitment(request),
            "started_at": datetime.now(UTC).isoformat(),
        }
        try:
            response = await self._delegate.acall(
                messages,
                tools=tools,
                output_model=output_model,
                **kwargs,
            )
        except Exception as error:
            self._budget.stop_reason = f"provider_error:{type(error).__name__}"
            self._budget.persist()
            _append_jsonl(
                self._trace_path,
                {
                    **base,
                    "status": "error",
                    "elapsed_seconds": time.monotonic() - started,
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
            )
            raise
        usage = response.usage if isinstance(response.usage, dict) else {}
        raw_cost = usage.get("cost")
        cost = (
            float(raw_cost)
            if isinstance(raw_cost, (int, float)) and not isinstance(raw_cost, bool)
            else None
        )
        provider_route = _provider_route(response.raw_response)
        record = {
            **base,
            "status": "complete",
            "elapsed_seconds": time.monotonic() - started,
            "usage": _jsonable(usage),
            "provider_route": provider_route,
            "finish_reason": response.finish_reason,
            "content": _jsonable(response.content),
            "reasoning": _jsonable(response.reasoning),
            "tool_calls": [
                {
                    "id": call.id,
                    "name": call.name,
                    "arguments": _normalized_arguments(call.arguments),
                }
                for call in response.tool_calls
            ],
        }
        _append_jsonl(self._trace_path, record)
        self._budget.observe_cost(cost)
        if provider_route.get("reported_provider") != EXPECTED_PROVIDER:
            self._budget.stop_reason = "unexpected_provider_route"
            self._budget.persist()
            raise BudgetStop(self._budget.stop_reason)
        return response


async def main(*, preflight_only: bool) -> None:
    _validate_environment()
    before_manifest = _read_json(_call_path(SOURCE_CALL) / "workspace.json")
    candidate_manifest = _read_json(_call_path(CANDIDATE_CALL) / "workspace.json")
    if _workspace_sha(before_manifest, "world_model.py") != CANONICAL_SHA:
        raise RuntimeError("canonical source commitment drifted")
    if _workspace_sha(candidate_manifest, "world_model.py") != CANDIDATE_SHA:
        raise RuntimeError("retained source commitment drifted")

    live_llm, kind = build_llm()
    if kind != "live" or live_llm.model != MODEL:
        raise RuntimeError(f"provider/model drift: {kind} {live_llm.model}")
    source_request = _read_json(_call_path(SOURCE_CALL) / "request.json")
    replay_context = _replay_context(live_llm)
    if replay_context != source_request["replay_context"]:
        raise RuntimeError("provider replay context drifted")

    if preflight_only:
        with tempfile.TemporaryDirectory(prefix="builder-continuation-") as temporary:
            root = Path(temporary)
            preparations = [
                _prepare_arm(root / arm, arm, before_manifest) for arm in ORDER
            ]
            manifest = _manifest(
                before_manifest,
                candidate_manifest,
                replay_context,
                preparations,
            )
            print(json.dumps({"status": "preflight_pass", **manifest}, indent=2))
        await live_llm.aclose()
        return

    arms_root = OUTPUT / "arms"
    if arms_root.exists() or (OUTPUT / "result.json").exists():
        raise FileExistsError("refusing to overwrite matched continuation evidence")
    arms_root.mkdir()
    preparations = [
        _prepare_arm(arms_root / arm, arm, before_manifest) for arm in ORDER
    ]
    manifest = _manifest(
        before_manifest,
        candidate_manifest,
        replay_context,
        preparations,
    )
    _write_json(OUTPUT / "manifest.json", manifest)
    budget = BudgetState(OUTPUT / "budget.json")
    budget.persist()
    provider_trace = OUTPUT / "provider-calls.jsonl"
    arm_results = []
    try:
        for arm in ORDER:
            arm_results.append(
                await _run_arm(
                    arms_root / arm,
                    arm,
                    live_llm,
                    budget,
                    provider_trace,
                )
            )
    finally:
        await live_llm.aclose()
    result = _assess(manifest, arm_results, budget, provider_trace)
    _write_json(OUTPUT / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "complete":
        raise SystemExit(2)


def _prepare_arm(root: Path, arm: str, before_manifest: dict[str, Any]) -> dict[str, Any]:
    workspace_root = root / "workspace"
    workspace = workspace_root / "ar25"
    _materialize(before_manifest, workspace)
    starting_sha = STARTING_MODELS[arm]
    (workspace / "world_model.py").write_bytes(_blob(starting_sha).read_bytes())
    if _sha256(workspace / "world_model.py") != starting_sha:
        raise RuntimeError(f"failed to prepare {arm} starting model")
    turn = _turn18()
    agent = TychoNooaAgent(
        llm=FakeLLMClient(),
        game_id="ar25",
        workspace_root=workspace_root,
        available_actions=turn.available_actions,
        resume_workspace=True,
    )
    builder = FocusedBuilder(
        llm=FakeLLMClient(),
        executor=agent._tool_executor,
        trace_path=None,
        config=BuilderConfig(max_tool_steps=MAX_TOOL_STEPS, max_tokens=MAX_TOKENS),
    )
    verifier_text, _ = verify_state(agent._tool_executor)
    baseline = _metrics(verifier_text)
    expected = {
        "canonical": (0, 16),
        "retained_candidate": (8, 16),
    }[arm]
    if (baseline["matched"], baseline["evaluated"]) != expected:
        raise RuntimeError(f"{arm} baseline drifted: {baseline}")
    return {
        "arm": arm,
        "workspace_root": str(workspace_root),
        "tree_commitment": _tree_commitment(workspace),
        "starting_model_sha256": starting_sha,
        "evidence_frontier_commitment": _frontier_commitment(workspace),
        "verifier_sha256": _sha256(workspace / "verify.py"),
        "baseline_verifier_metrics": baseline,
        "builder_system_prompt_commitment": _commitment(
            builder._resolve_system_prompt()
        ),
        "builder_tools_commitment": _commitment(
            _tool_schemas(builder._tool_surface.native_tools())
        ),
    }


async def _run_arm(
    root: Path,
    arm: str,
    delegate: UnifiedLLM,
    budget: BudgetState,
    provider_trace: Path,
) -> dict[str, Any]:
    workspace_root = root / "workspace"
    workspace = workspace_root / "ar25"
    turn = _turn18()
    capped = BudgetedLLM(
        delegate,
        arm=arm,
        budget=budget,
        trace_path=provider_trace,
    )
    agent = TychoNooaAgent(
        llm=capped,
        game_id="ar25",
        workspace_root=workspace_root,
        available_actions=turn.available_actions,
        tool_trace_path=root / "actor-tool-trace.jsonl",
        resume_workspace=True,
    )
    builder = FocusedBuilder(
        llm=capped,
        executor=agent._tool_executor,
        trace_path=root / "builder-tool-trace.jsonl",
        config=BuilderConfig(max_tool_steps=MAX_TOOL_STEPS, max_tokens=MAX_TOKENS),
    )
    llm_trace = LLMTrace(root / "llm-trace.jsonl")
    llm_trace.attach(builder.event_manager, source=f"builder:{arm}")
    before_text, _ = verify_state(agent._tool_executor)
    (root / "verifier-before.txt").write_text(before_text, encoding="utf-8")
    before_metrics = _metrics(before_text)
    status = "complete"
    error = None
    builder_result = None
    started = time.monotonic()
    try:
        async with asyncio.timeout(MAX_SECONDS_PER_ARM):
            built = await builder.build(
                BuilderRequest(
                    turn=turn,
                    reason=(
                        "Continue mechanics repair at the preserved turn-18 frontier. "
                        "Find the first graded transition still simulated wrong and "
                        "improve exact transition accuracy without changing the task."
                    ),
                )
            )
        builder_result = {
            "report": built.report,
            "report_fresh": built.report_fresh,
            "report_forced": built.report_forced,
            "world_model_checkpoint_forced": built.world_model_checkpoint_forced,
            "artifact_hashes": built.artifact_hashes,
        }
    except TimeoutError as caught:
        status = "timeout"
        error = f"{type(caught).__name__}: {caught}"
    except BudgetStop as caught:
        status = "budget_stop"
        error = str(caught)
    except Exception as caught:  # noqa: BLE001 - preserve provider/tool failure evidence
        status = "error"
        error = f"{type(caught).__name__}: {caught}"
    after_text, _ = verify_state(agent._tool_executor)
    (root / "verifier-after.txt").write_text(after_text, encoding="utf-8")
    calls = _jsonl_or_empty(root / "builder-tool-trace.jsonl")
    provider_calls = [
        row for row in _jsonl_or_empty(provider_trace) if row.get("arm") == arm
    ]
    trajectory = _score_trajectory(calls, provider_calls)
    result = {
        "arm": arm,
        "status": status,
        "error": error,
        "elapsed_seconds": time.monotonic() - started,
        "starting_model_sha256": STARTING_MODELS[arm],
        "final_model_sha256": _sha256(workspace / "world_model.py"),
        "before": before_metrics,
        "after": _metrics(after_text),
        "score_gain": _score_gain(before_metrics, _metrics(after_text)),
        "provider_calls": len(provider_calls),
        "observed_cost_usd": round(
            sum(
                float(cost)
                for row in provider_calls
                if isinstance(row.get("usage"), dict)
                and isinstance((cost := row["usage"].get("cost")), (int, float))
                and not isinstance(cost, bool)
            ),
            12,
        ),
        "tool_calls": len(calls),
        "score_trajectory": trajectory,
        "builder_result": builder_result,
        "workspace_commitment_after": _tree_commitment(workspace),
    }
    _write_json(root / "result.json", result)
    return result


def _assess(
    manifest: dict[str, Any],
    arms: list[dict[str, Any]],
    budget: BudgetState,
    provider_trace: Path,
) -> dict[str, Any]:
    by_arm = {row["arm"]: row for row in arms}
    routes = {
        _canonical_json(row.get("provider_route", {}))
        for row in _jsonl_or_empty(provider_trace)
        if row.get("status") == "complete"
    }
    invalid_reasons = []
    if set(by_arm) != set(ORDER):
        invalid_reasons.append("an arm did not execute")
    if any(row["status"] != "complete" for row in arms):
        invalid_reasons.append("an arm did not complete")
    if budget.stop_reason is not None:
        invalid_reasons.append(f"budget/provider stop: {budget.stop_reason}")
    if len(routes) > 1:
        invalid_reasons.append("observed provider route differed across calls")
    preparations = {row["arm"]: row for row in manifest["arms"]}
    parity_fields = (
        "evidence_frontier_commitment",
        "verifier_sha256",
        "builder_system_prompt_commitment",
        "builder_tools_commitment",
    )
    for parity_field in parity_fields:
        if len({row[parity_field] for row in preparations.values()}) != 1:
            invalid_reasons.append(f"arm {parity_field} differed")
    judgment = "invalid"
    threshold = None
    threshold_calls = None
    if not invalid_reasons:
        retained = by_arm["retained_candidate"]
        canonical = by_arm["canonical"]
        retained_final = retained["after"]["accuracy"]
        canonical_final = canonical["after"]["accuracy"]
        if retained_final > canonical_final:
            judgment = "retained_better"
        elif canonical_final > retained_final:
            judgment = "canonical_better"
        else:
            threshold = retained_final
            threshold_calls = {
                arm: _first_provider_call_at_or_above(by_arm[arm], threshold)
                for arm in ORDER
            }
            if threshold_calls["retained_candidate"] < threshold_calls["canonical"]:
                judgment = "retained_faster"
            elif threshold_calls["canonical"] < threshold_calls["retained_candidate"]:
                judgment = "canonical_faster"
            else:
                judgment = "equal_inconclusive"
    return {
        "schema": "tycho-nooa-matched-builder-continuation-v1",
        "status": "complete" if not invalid_reasons else "invalid",
        "manifest_commitment": _commitment(manifest),
        "arms": arms,
        "observed_provider_routes": [json.loads(route) for route in sorted(routes)],
        "budget": _read_json(budget.path),
        "invalid_reasons": invalid_reasons,
        "judgment": judgment,
        "shared_final_threshold": threshold,
        "first_provider_call_to_shared_threshold": threshold_calls,
        "claim_ceiling": (
            "one matched shadow continuation at the preserved turn-18 frontier; "
            "no ARC action, gameplay consequence, replication, or transfer claim"
        ),
    }


def _manifest(
    before_manifest: dict[str, Any],
    candidate_manifest: dict[str, Any],
    replay_context: dict[str, Any],
    arms: list[dict[str, Any]],
) -> dict[str, Any]:
    differences = _manifest_differences(before_manifest, candidate_manifest)
    if differences != ["world_model.py"]:
        raise RuntimeError(f"source workspace difference is not isolated: {differences}")
    if len({row["evidence_frontier_commitment"] for row in arms}) != 1:
        raise RuntimeError("prepared evidence frontiers differ")
    parity_fields = (
        "verifier_sha256",
        "builder_system_prompt_commitment",
        "builder_tools_commitment",
    )
    for parity_field in parity_fields:
        if len({row[parity_field] for row in arms}) != 1:
            raise RuntimeError(f"prepared {parity_field} values differ")
    return {
        "schema": "tycho-nooa-matched-builder-continuation-manifest-v1",
        "experiment_card": str((OUTPUT / "EXPERIMENT-CARD.md").relative_to(ROOT)),
        "experiment_card_sha256": _sha256(OUTPUT / "EXPERIMENT-CARD.md"),
        "source_run": str(SOURCE.parent.relative_to(ROOT)),
        "source_call": SOURCE_CALL,
        "candidate_call": CANDIDATE_CALL,
        "source_workspace_difference": differences,
        "order_seed": 20260822,
        "order": list(ORDER),
        "model": MODEL,
        "provider_replay_context": replay_context,
        "provider_routing": {
            "order": [PROVIDER_ENDPOINT],
            "allow_fallbacks": False,
            "expected_reported_provider": EXPECTED_PROVIDER,
        },
        "builder_config": {
            "max_tool_steps": MAX_TOOL_STEPS,
            "max_tokens": MAX_TOKENS,
            "temperature": None,
            "max_seconds_per_arm": MAX_SECONDS_PER_ARM,
        },
        "budget": {
            "max_calls_per_arm": MAX_CALLS_PER_ARM,
            "max_calls_total": MAX_CALLS_TOTAL,
            "max_cost_usd": MAX_COST_USD,
            "reserve_per_call_usd": RESERVE_PER_CALL_USD,
        },
        "task": (
            "Continue mechanics repair at preserved turn 18; improve exact transition "
            "accuracy from the first remaining divergence."
        ),
        "arms": arms,
        "execution": {"arc_environment_calls": 0},
    }


def _turn18() -> GameTurn:
    actions = [
        row
        for row in _jsonl_or_empty(SOURCE / "transcript.jsonl")
        if row.get("type") == "action_result"
    ]
    current = next(row for row in actions if row.get("sequence") == 18)
    previous = next(row for row in actions if row.get("sequence") == 17)
    before = current["before"]
    decision = previous["decision"]
    return GameTurn(
        grid=before["decision_frame"],
        available_actions=before["available_actions"],
        sequence=18,
        level=0,
        turn_in_level=18,
        levels_completed=before["levels_completed"],
        state=before["state"],
        previous_action=decision["action"],
        previous_row=decision.get("row"),
        previous_col=decision.get("col"),
    )


def _score_trajectory(
    tool_calls: list[dict[str, Any]], provider_calls: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    provider_by_tool = {
        tool["id"]: row["arm_call_index"]
        for row in provider_calls
        for tool in row.get("tool_calls", [])
    }
    trajectory = []
    for row in tool_calls:
        metrics = _metrics(str(row.get("result", "")))
        if metrics["accuracy"] is None:
            continue
        trajectory.append(
            {
                "provider_call_index": provider_by_tool.get(row.get("tool_call_id")),
                "tool_call_index": row.get("call_index"),
                "tool": row.get("name"),
                **metrics,
            }
        )
    return trajectory


def _metrics(text: str) -> dict[str, Any]:
    accuracy = ACCURACY.search(text) or ACCURACY_AUTO.search(text)
    replay = REPLAY.search(text) or REPLAY_AUTO.search(text)
    divergence = FIRST_DIVERGENCE.search(text)
    return {
        "accuracy": float(accuracy.group(1)) if accuracy is not None else None,
        "matched": int(replay.group(1)) if replay is not None else None,
        "evaluated": int(replay.group(2)) if replay is not None else None,
        "observed": int(replay.group(3)) if replay is not None else None,
        "first_divergence": divergence.group(1) if divergence is not None else None,
    }


def _score_gain(before: dict[str, Any], after: dict[str, Any]) -> float | None:
    if before["accuracy"] is None or after["accuracy"] is None:
        return None
    return round(after["accuracy"] - before["accuracy"], 12)


def _first_provider_call_at_or_above(arm: dict[str, Any], threshold: float) -> int:
    if arm["before"]["accuracy"] >= threshold:
        return 0
    matches = [
        row["provider_call_index"]
        for row in arm["score_trajectory"]
        if row["provider_call_index"] is not None and row["accuracy"] >= threshold
    ]
    return min(matches) if matches else arm["provider_calls"] + 1


def _validate_environment() -> None:
    expected = {
        "ARC_LLM_MODEL": "google/gemma-4-31b-it",
        "ARC_LLM_CONTEXT_WINDOW": "262144",
        "ARC_LLM_MAX_TOKENS": "4096",
        "ARC_LLM_USE_RESPONSES": "0",
        "TYCHO_NOOA_REQUEST_REASONING": "0",
        "TYCHO_SANDBOX_RUNTIME": "host",
    }
    drift = {
        name: os.environ.get(name)
        for name, value in expected.items()
        if os.environ.get(name) != value
    }
    if drift:
        raise RuntimeError(f"experiment environment drift: {drift}")
    if not os.environ.get("ARC_LLM_BASE_URL") or not os.environ.get("ARC_LLM_API_KEY"):
        raise RuntimeError("gateway configuration is absent")
    if len((OUTPUT / "EXPERIMENT-CARD.md").read_bytes()) > 1024:
        raise RuntimeError("experiment card exceeds 1 KiB")


def _materialize(manifest: dict[str, Any], destination: Path) -> None:
    destination.mkdir(parents=True)
    for directory in manifest["directories"]:
        path = destination / directory["path"]
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(directory["mode"])
    for entry in manifest["files"]:
        path = destination / entry["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_blob(entry["sha256"]).read_bytes())
        path.chmod(entry["mode"])
        if _sha256(path) != entry["sha256"]:
            raise RuntimeError(f"blob commitment mismatch: {entry['path']}")


def _manifest_differences(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    left = {row["path"]: row["sha256"] for row in a["files"]}
    right = {row["path"]: row["sha256"] for row in b["files"]}
    return sorted(path for path in left.keys() | right.keys() if left.get(path) != right.get(path))


def _frontier_commitment(workspace: Path) -> str:
    rows = [
        {"path": path.relative_to(workspace).as_posix(), "sha256": _sha256(path)}
        for path in sorted(workspace.rglob("turn_*.json"))
    ]
    return _commitment(rows)


def _tree_commitment(workspace: Path) -> str:
    rows = [
        {"path": path.relative_to(workspace).as_posix(), "sha256": _sha256(path)}
        for path in sorted(workspace.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    ]
    return _commitment(rows)


def _tool_schemas(tools: list[Tool]) -> list[dict[str, Any]]:
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.get_parameter_schema(),
        }
        for tool in tools
    ]


def _normalized_arguments(value: Any) -> Any:
    if not isinstance(value, str):
        return _jsonable(value)
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {"malformed_json": value}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    return str(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _commitment(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _workspace_sha(manifest: dict[str, Any], relative: str) -> str:
    matches = [row["sha256"] for row in manifest["files"] if row["path"] == relative]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {relative} commitment")
    return matches[0]


def _call_path(call: str) -> Path:
    return SOURCE / "decision-checkpoints/calls" / call


def _blob(sha256: str) -> Path:
    path = SOURCE / "decision-checkpoints/blobs/sha256" / sha256
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _jsonl_or_empty(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(_canonical_json(value) + "\n")
        stream.flush()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    asyncio.run(main(preflight_only=arguments.preflight_only))
