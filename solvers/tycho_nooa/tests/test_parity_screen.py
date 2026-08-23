from __future__ import annotations

from pathlib import Path

import pytest
from nooa.unifiedllm import FakeLLMClient, LLMResponse

from tycho_nooa.parity_manifest import ExperimentCaps, ModelRoute
from tycho_nooa.parity_screen import (
    BudgetState,
    CappedParityLLM,
    _validate_configured_model,
    _validate_observed_route,
    execute_authorized_screen,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MANIFEST = REPOSITORY_ROOT / "analysis/codeact-cognitive-parity/v1/experiment.json"


async def test_disabled_manifest_stops_before_output_or_provider_call(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "run"
    native = FakeLLMClient()
    codeact = FakeLLMClient()

    with pytest.raises(RuntimeError, match="execution is disabled"):
        await execute_authorized_screen(
            MANIFEST,
            repository_root=REPOSITORY_ROOT,
            destination=destination,
            native_llm=native,
            codeact_llm=codeact,
        )

    assert not destination.exists()
    assert native.call_count == 0
    assert codeact.call_count == 0


def test_budget_reserves_before_call_and_rejects_missing_cost(tmp_path: Path) -> None:
    budget = BudgetState(
        ExperimentCaps(
            sessions=2,
            calls_per_session=2,
            calls_total=2,
            output_tokens_per_call=100,
            wall_seconds_per_session=10,
            max_cost_usd=0.2,
            reserve_per_call_usd=0.1,
        ),
        path=tmp_path / "budget.json",
    )

    assert budget.reserve() == 1
    with pytest.raises(RuntimeError, match="lacks valid observed cost"):
        budget.observe(_response({"prompt_tokens": 1}))
    with pytest.raises(RuntimeError, match="budget is stopped"):
        budget.reserve()
    assert budget.stop_reason == "missing_cost_evidence"
    assert (tmp_path / "budget.json").is_file()


def test_budget_refuses_call_that_cannot_be_reserved(tmp_path: Path) -> None:
    budget = BudgetState(
        ExperimentCaps(
            sessions=2,
            calls_per_session=2,
            calls_total=2,
            output_tokens_per_call=100,
            wall_seconds_per_session=10,
            max_cost_usd=0.15,
            reserve_per_call_usd=0.1,
        ),
        path=tmp_path / "budget.json",
    )

    budget.reserve()
    budget.observe(_response({"cost": 0.06}))
    with pytest.raises(RuntimeError, match="cost reservation refused"):
        budget.reserve()

    assert budget.calls == 1


async def test_capped_client_records_matched_route_and_observed_cost(
    tmp_path: Path,
) -> None:
    route = ModelRoute(
        model="fake-model",
        provider="provider-a",
        reasoning_effort="high",
        temperature=0,
    )
    delegate = FakeLLMClient(
        scripted_responses=[
            _response(
                {"prompt_tokens": 3, "completion_tokens": 2, "cost": 0.01},
                raw_response={"provider": "provider-a", "model": "fake-model"},
            )
        ]
    )
    delegate.config.update(reasoning_effort="high", temperature=0)
    budget = _budget(tmp_path)
    subject = CappedParityLLM(
        delegate,
        arm="native",
        expected_route=route,
        budget=budget,
        trace_path=tmp_path / "provider.jsonl",
    )

    response = await subject.acall([], tools=[])

    assert response.finish_reason == "stop"
    assert budget.calls == 1
    assert budget.observed_cost_usd == 0.01
    trace = (tmp_path / "provider.jsonl").read_text()
    assert '"status":"complete"' in trace
    assert '"provider":"provider-a"' in trace
    with pytest.raises(RuntimeError, match="async provider calls"):
        subject.call([])


async def test_capped_client_stops_after_observed_route_drift(tmp_path: Path) -> None:
    route = ModelRoute(
        model="fake-model",
        provider="provider-a",
        reasoning_effort="high",
    )
    delegate = FakeLLMClient(
        scripted_responses=[
            _response(
                {"cost": 0.01},
                raw_response={"provider": "provider-b", "model": "fake-model"},
            )
        ]
    )
    subject = CappedParityLLM(
        delegate,
        arm="codeact",
        expected_route=route,
        budget=_budget(tmp_path),
        trace_path=tmp_path / "provider.jsonl",
    )

    with pytest.raises(RuntimeError, match="provider route drift"):
        await subject.acall([])

    assert '"status":"error"' in (tmp_path / "provider.jsonl").read_text()


def test_configured_and_observed_route_checks_are_factored() -> None:
    route = ModelRoute(
        model="fake-model",
        provider="provider-a",
        reasoning_effort="high",
        temperature=0,
    )
    llm = FakeLLMClient()
    llm.config.update(reasoning={"effort": "high"}, temperature=0)
    _validate_configured_model(llm, route)
    _validate_observed_route({"provider": "provider-a", "model": "fake-model"}, route)

    llm.model = "wrong"
    with pytest.raises(ValueError, match="configured model"):
        _validate_configured_model(llm, route)
    llm.model = "fake-model"
    llm.config["reasoning"] = {"effort": "low"}
    with pytest.raises(ValueError, match="reasoning effort"):
        _validate_configured_model(llm, route)
    llm.config["reasoning"] = {"effort": "high"}
    llm.config["temperature"] = 1
    with pytest.raises(ValueError, match="temperature"):
        _validate_configured_model(llm, route)
    with pytest.raises(RuntimeError, match="model route drift"):
        _validate_observed_route({"provider": "provider-a", "model": "wrong"}, route)


def _budget(tmp_path: Path) -> BudgetState:
    return BudgetState(
        ExperimentCaps(
            sessions=2,
            calls_per_session=2,
            calls_total=2,
            output_tokens_per_call=100,
            wall_seconds_per_session=10,
            max_cost_usd=0.2,
            reserve_per_call_usd=0.1,
        ),
        path=tmp_path / "budget.json",
    )


def _response(usage: dict, *, raw_response=None) -> LLMResponse:
    return LLMResponse(
        raw_response=raw_response,
        content="",
        tool_calls=[],
        finish_reason="stop",
        assistant_message={"role": "assistant", "content": ""},
        usage=usage,
    )
