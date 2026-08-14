from __future__ import annotations

import json
from pathlib import Path

import pytest
from nooa import GenerationError
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

import launcher
import provider as provider_module
from agent import (
    CellMaskModel,
    GroundingAgent,
    GroundingClaimDraftModel,
    ProbeCommitmentDraftModel,
)
from launcher import LIVE_PROVIDER_ROUTE, _client_for, _tail_from_start
from provider import AuditedProviderClient


def response(code: str, *, usage: dict[str, int] | None = None) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[ToolCall(id="fixture", name="execute_python", arguments=json.dumps({"code": code}))],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": ""},
        usage=usage,
    )


def artifact_input() -> dict:
    frame = {"evidence_id": "frame-a", "kind": "frame", "grid": [[0, 7]], "available_actions": [1, 2]}
    return {
        "phase": "claim",
        "current_observation_id": "frame-a",
        "legal_actions": [1, 2],
        "accepted_claim": None,
        "evidence": {
            "frames": [frame], "frame_summaries": [{key: value for key, value in frame.items() if key != "grid"}],
            "transitions": [], "probe_results": [], "attempts": [], "events": {}, "commitment": "e",
        },
        "model_contract": {"schema": "fixture-public-contract"},
        "run_policy": {
            "mode": "review_stop",
            "ordinary_cycle_stop": "terminate_run",
            "g2": "stop_for_review",
        },
    }


def commitment_input() -> dict:
    value = artifact_input()
    value["phase"] = "commitment"
    value["accepted_claim"] = {
        "claim_id": "claim-a",
        "observation_id": "frame-a",
        "primary_region": {
            "bbox": {"row": 0, "col": 1, "height": 1, "width": 1},
            "rows": ["#"],
        },
        "alternatives": [],
        "evidence_refs": ["frame-a"],
        "parent_claim_id": None,
    }
    return value


def post_probe_claim_input(
    *, outside_primary_changed: bool = True, with_prior_heuristic: bool = False
) -> dict:
    value = artifact_input()
    value["current_observation_id"] = "frame-b"
    frame = {
        "evidence_id": "frame-b",
        "kind": "frame",
        "grid": [[0, 7]],
        "available_actions": [1, 2],
    }
    value["evidence"]["frames"] = [frame]
    value["evidence"]["frame_summaries"] = [
        {key: item for key, item in frame.items() if key != "grid"}
    ]
    probe_result = {
            "evidence_id": "probe-result-a",
            "after_id": "frame-b",
            "change_accounting": {
                "outside_primary_changed": outside_primary_changed,
                "changed_cells_outside_primary": 52 if outside_primary_changed else 0,
            },
        }
    if with_prior_heuristic:
        probe_result["grounding_claim"] = {
            "heuristic_register": [
                {
                    "name": "background_progression",
                    "belief": "The background pattern may advance once per elapsed action.",
                    "scope": "One tested transition.",
                    "evidence_refs": ["frame-a"],
                    "decision_role": "watch",
                    "reconsider_when": "A later step does not advance it.",
                    "epistemic_status": "provisional",
                }
            ]
        }
    value["evidence"]["probe_results"] = [probe_result]
    return value


def test_cell_mask_schema_exposes_encoding_and_rejects_failed_live_artifact() -> None:
    schema = CellMaskModel.model_json_schema()
    rows = schema["properties"]["rows"]
    assert rows["items"]["pattern"] == "^[#.]+$"
    assert "# selects a cell" in rows["items"]["description"]
    assert "exactly bbox.width characters" in rows["description"]

    failed_live_mask = {
        "bbox": {"row": 45, "col": 34, "height": 2, "width": 5},
        "rows": ["12, 12, 12, 12, 12", "12, 12, 12, 12, 12"],
    }
    with pytest.raises(ValueError, match="string_pattern_mismatch"):
        CellMaskModel.model_validate(failed_live_mask)


def test_claim_schema_allows_no_alternative_and_rejects_duplicate_live_artifact() -> None:
    primary = {
        "bbox": {"row": 61, "col": 13, "height": 2, "width": 42},
        "rows": ["#" * 42, "#" * 42],
    }
    base = {
        "observation_id": "frame-a",
        "primary_region": primary,
        "evidence_refs": ["frame-a"],
        "parent_claim_id": None,
    }
    accepted = GroundingClaimDraftModel.model_validate({**base, "alternatives": []})
    assert accepted.alternatives == []

    with pytest.raises(ValueError, match="must all be distinct"):
        GroundingClaimDraftModel.model_validate({**base, "alternatives": [primary]})

    alternatives_schema = GroundingClaimDraftModel.model_json_schema()["properties"]["alternatives"]
    assert alternatives_schema["maxItems"] == 3
    assert "Use []" in alternatives_schema["description"]


def test_runtime_commitment_union_has_exact_host_parity() -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from host.protocol import predicted_effect_from_record

    records = [
        {"kind": "translate", "source": "primary_region", "d_row": -1, "d_col": 0},
        {"kind": "unchanged", "source": "primary_region"},
        {
            "kind": "replace",
            "source": "primary_region",
            "expected_after": {
                "bbox": {"row": 1, "col": 1, "height": 1, "width": 2},
                "rows": ["##"],
            },
            "assertions": [{"row": 1, "col": 1, "value": 7}],
        },
    ]
    for effect in records:
        alternative_effect = (
            {"kind": "translate", "source": "primary_region", "d_row": 1, "d_col": 0}
            if effect["kind"] == "unchanged"
            else {"kind": "unchanged", "source": "primary_region"}
        )
        draft = ProbeCommitmentDraftModel.model_validate(
            {
                "claim_id": "claim-a",
                "action": 1,
                "data": None,
                "predicted_effect": effect,
                "alternative_predictions": [
                    {"action": 2, "data": None, "predicted_effect": alternative_effect}
                ],
            }
        )
        dumped = draft.model_dump(mode="json")["predicted_effect"]
        assert predicted_effect_from_record(dumped).to_record() == effect

    with pytest.raises(ValueError, match="union_tag_not_found"):
        ProbeCommitmentDraftModel.model_validate(
            {
                "claim_id": "claim-a",
                "action": 1,
                "data": None,
                "predicted_effect": {
                    "type": "bounded_replace",
                    "primary_region_anchor": {},
                    "replacement": ["#"],
                },
                "alternative_predictions": [
                    {
                        "action": 2,
                        "data": None,
                        "predicted_effect": {"kind": "unchanged", "source": "primary_region"},
                    }
                ],
            }
        )


@pytest.mark.asyncio
async def test_agent_returns_typed_claim_from_fresh_empirical_surface(tmp_path: Path) -> None:
    client = FakeLLMClient()
    client._response_queue.append(
        response(
                "state = inspect_evidence()\n"
                "assert state['visible_frame_window'][0]['evidence_id'] == 'frame-a'\n"
                "assert state['current_frame']['evidence_id'] == 'frame-a'\n"
                "assert state['frame_summaries'][0]['evidence_id'] == 'frame-a'\n"
                "assert state['probe_results'] == ()\n"
                "assert state['current_transition'] is None\n"
                "assert state['current_probe_result'] is None\n"
                "try:\n    artifact_input\nexcept NameError:\n    pass\nelse:\n    raise AssertionError('raw host input leaked')\n"
                "try:\n    self\nexcept NameError:\n    pass\nelse:\n    raise AssertionError('self leaked')\n"
                "return_result({'observation_id':'frame-a',"
                "'primary_region':{'bbox':{'row':0,'col':1,'height':1,'width':1},'rows':['#']},"
                "'alternatives':[{'bbox':{'row':0,'col':0,'height':1,'width':1},'rows':['#']}],"
                "'evidence_refs':['frame-a'],'parent_claim_id':None})"
            )
        )
    agent = GroundingAgent(llm=client, workspace=str(tmp_path / "workspace"))
    result = await agent.produce(artifact_input())
    assert result["observation_id"] == "frame-a"
    assert result["primary_region"]["rows"] == ["#"]


@pytest.mark.asyncio
async def test_model_prompt_explains_semantic_mechanical_boundary_and_examples(tmp_path: Path) -> None:
    class CapturingClient(FakeLLMClient):
        def __init__(self) -> None:
            super().__init__()
            self.messages = []

        async def acall(self, messages, *args, **kwargs):
            self.messages = messages
            return await super().acall(messages, *args, **kwargs)

    client = CapturingClient()
    client._response_queue.append(
        response(
            "return_result(submit_grounding_claim("
            "primary_region=make_mask(0,1,['#']), alternatives=[]))"
        )
    )
    await GroundingAgent(llm=client, workspace=str(tmp_path / "prompt-boundary")).produce(
        artifact_input()
    )
    system = client.messages[0]["content"]
    assert "WHY THE HARNESS EXISTS" in system
    assert "Do the semantic work deterministic code cannot do" in system
    assert "host does mechanical work" in system
    assert "inspect_evidence()['current_frame']" in system
    assert "submit_grounding_claim() and commit_contrastive() submit immediately" in system
    assert "exactly 10 execute_python calls" in system
    assert "already bound Python globals" in system


@pytest.mark.asyncio
async def test_commitment_prompt_exposes_exact_relative_variadic_cell_form(tmp_path: Path) -> None:
    class CapturingClient(FakeLLMClient):
        def __init__(self) -> None:
            super().__init__()
            self.messages = []

        async def acall(self, messages, *args, **kwargs):
            self.messages = messages
            return await super().acall(messages, *args, **kwargs)

    client = CapturingClient()
    client._response_queue.append(
        response(
            "selected = predict_cell_values(cell_after(d_row=0, d_col=0, value=7))\n"
            "contrast = action_prediction(2, predict_unchanged())\n"
            "return_result(commit_contrastive(1, selected, [contrast]))"
        )
    )
    await GroundingAgent(
        llm=client, workspace=str(tmp_path / "relative-cell-prompt")
    ).produce(commitment_input())
    system = client.messages[0]["content"]
    assert "predict_cell_values(cell_after(d_row=1, d_col=1, value=3))" in system
    assert "offsets from the top-left of accepted_claim.primary_region" in system
    assert "with no list wrapper" in system
    assert "predict_replace" not in system


@pytest.mark.asyncio
async def test_post_probe_claim_requires_and_preserves_model_authored_update(
    tmp_path: Path,
) -> None:
    client = FakeLLMClient()
    client._response_queue.append(
        response(
            "update = account_prior(disposition='abandon', "
            "rationale='The prior primary stayed fixed while change occurred elsewhere.')\n"
            "h = provisional_heuristic(name='candidate_motion', "
            "belief='A different region may respond to the action.', "
            "scope='One observed transition.', "
            "decision_role='active', reconsider_when='Its prospective effect mismatches.')\n"
            "primary = make_mask(0, 1, ['#'])\n"
            "return_result(submit_grounding_claim(primary, [], hypothesis_update=update, "
            "heuristic_updates=[h]))"
        )
    )
    result = await GroundingAgent(
        llm=client, workspace=str(tmp_path / "post-probe-accounting")
    ).produce(post_probe_claim_input())
    assert result["hypothesis_update"] == {
        "prior_probe_result_id": "probe-result-a",
        "disposition": "abandon",
        "rationale": "The prior primary stayed fixed while change occurred elsewhere.",
    }
    assert result["heuristic_updates"][0]["name"] == "candidate_motion"
    assert "evidence_refs" not in result["heuristic_updates"][0]


@pytest.mark.asyncio
async def test_evidence_conservation_prompt_delegates_fact_to_host_and_judgment_to_model(
    tmp_path: Path,
) -> None:
    class CapturingClient(FakeLLMClient):
        def __init__(self) -> None:
            super().__init__()
            self.messages = []

        async def acall(self, messages, *args, **kwargs):
            self.messages = messages
            return await super().acall(messages, *args, **kwargs)

    client = CapturingClient()
    client._response_queue.append(
        response(
            "update = account_prior(disposition='revise', "
            "rationale='The interpretation remains scoped and revisable.')\n"
            "h = provisional_heuristic(name='background_progression', "
            "belief='The other change may recur across chosen actions.', "
            "scope='One observed transition.', "
            "decision_role='watch', reconsider_when='The progression differs on a later step.')\n"
            "return_result(submit_grounding_claim(make_mask(0,1,['#']), [], "
            "hypothesis_update=update, heuristic_updates=[h]))"
        )
    )
    await GroundingAgent(
        llm=client, workspace=str(tmp_path / "evidence-conservation-prompt")
    ).produce(post_probe_claim_input())
    system = client.messages[0]["content"]
    assert "treating every interpretation as a scoped, revisable heuristic" in system
    assert "none may be called explained, closed, confirmed, or certain" in system
    assert "host preserves observations and checks only temporal/evidence binding" in system
    assert "you decide grouping" in system
    assert "relevance, identity, causal role" in system


@pytest.mark.asyncio
async def test_sandbox_accepts_changed_heuristic_without_reproducing_prior_register(
    tmp_path: Path,
) -> None:
    client = FakeLLMClient()
    common = (
        "update = account_prior(disposition='revise', "
        "rationale='The register remains provisional.')\n"
    )
    client._response_queue.append(
        response(
            common
            + "h = provisional_heuristic(name='replacement', belief='Another pattern may move.', "
            "scope='One transition.', decision_role='active', "
            "reconsider_when='Its prospective effect mismatches.')\n"
            "return_result(submit_grounding_claim(make_mask(0,1,['#']), [], "
            "hypothesis_update=update, heuristic_updates=[h]))"
        )
    )
    result = await GroundingAgent(
        llm=client,
        workspace=str(tmp_path / "heuristic-deletion-repair"),
        max_iterations=2,
        max_retries=2,
    ).produce(post_probe_claim_input(with_prior_heuristic=True))
    assert result["heuristic_updates"] == [
        {
            "name": "replacement",
            "belief": "Another pattern may move.",
            "scope": "One transition.",
            "decision_role": "active",
            "reconsider_when": "Its prospective effect mismatches.",
        }
    ]
    assert not client._response_queue


@pytest.mark.asyncio
async def test_long_run_prompt_explains_autonomous_revision_without_new_api(tmp_path: Path) -> None:
    class CapturingClient(FakeLLMClient):
        def __init__(self) -> None:
            super().__init__(); self.messages = []

        async def acall(self, messages, *args, **kwargs):
            self.messages = messages
            return await super().acall(messages, *args, **kwargs)

    value = artifact_input()
    value["run_policy"] = {
        "mode": "autonomous_44_action",
        "ordinary_cycle_stop": "record_and_reset_without_human",
        "g2": "record_reset_without_new_api",
    }
    client = CapturingClient()
    client._response_queue.append(
        response("return_result(submit_grounding_claim(make_mask(0,1,['#']), []))")
    )
    await GroundingAgent(llm=client, workspace=str(tmp_path / "long-run-prompt")).produce(value)
    system = client.messages[0]["content"]
    assert "reset without human input" in system
    assert "probe_results inside inspect_evidence()" in system
    assert "opens no new API" in system
    assert "Every execute_python call consumes one" in system
    assert "evidence remains readable after an error" in system
    assert "Do not print or restate a whole grid" in system


@pytest.mark.asyncio
async def test_atomic_commitment_helper_closes_effect_and_submission_in_one_call(
    tmp_path: Path,
) -> None:
    client = FakeLLMClient()
    client._response_queue.append(
        response(
            "selected = predict_translate(d_row=0, d_col=-1)\n"
            "contrast = action_prediction(action=2, predicted_effect=predict_unchanged())\n"
            "return_result(commit_contrastive("
            "action=1, predicted_effect=selected, alternative_predictions=[contrast], data=None))"
        )
    )
    agent = GroundingAgent(llm=client, workspace=str(tmp_path / "atomic-commitment"))
    result = await agent.produce(commitment_input())
    assert result == {
        "claim_id": "claim-a",
        "action": 1,
        "data": None,
        "predicted_effect": {
            "kind": "translate",
            "source": "primary_region",
            "d_row": 0,
            "d_col": -1,
        },
        "alternative_predictions": [
            {
                "action": 2,
                "data": None,
                "predicted_effect": {"kind": "unchanged", "source": "primary_region"},
            }
        ],
    }


@pytest.mark.asyncio
async def test_primary_relative_variadic_cells_expand_to_exact_host_effect(
    tmp_path: Path,
) -> None:
    value = commitment_input()
    value["accepted_claim"]["primary_region"] = {
        "bbox": {"row": 31, "col": 20, "height": 3, "width": 3},
        "rows": [".#.", "###", ".#."],
    }
    client = FakeLLMClient()
    client._response_queue.append(
        response(
            "selected = predict_cell_values(\n"
            "    cell_after(d_row=1, d_col=2, value=9),\n"
            "    cell_after(d_row=0, d_col=1, value=7),\n"
            "    cell_after(d_row=1, d_col=0, value=8),\n"
            ")\n"
            "contrast = action_prediction(2, predict_unchanged())\n"
            "return_result(commit_contrastive(1, selected, [contrast]))"
        )
    )
    result = await GroundingAgent(
        llm=client, workspace=str(tmp_path / "relative-cell-effect")
    ).produce(value)
    assert result["predicted_effect"] == {
        "kind": "replace",
        "source": "primary_region",
        "expected_after": {
            "bbox": {"row": 31, "col": 20, "height": 2, "width": 3},
            "rows": [".#.", "#.#"],
        },
        "assertions": [
            {"row": 31, "col": 21, "value": 7},
            {"row": 32, "col": 20, "value": 8},
            {"row": 32, "col": 22, "value": 9},
        ],
    }


@pytest.mark.asyncio
async def test_commitment_surface_omits_claim_and_old_replace_helpers(tmp_path: Path) -> None:
    client = FakeLLMClient()
    client._response_queue.append(
        response(
            "try:\n    make_mask\nexcept NameError:\n    pass\n"
            "else:\n    raise AssertionError('claim helper leaked')\n"
            "try:\n    predict_replace\nexcept NameError:\n    pass\n"
            "else:\n    raise AssertionError('old replace helper leaked')\n"
            "selected = predict_cell_values(cell_after(d_row=0, d_col=0, value=7))\n"
            "contrast = action_prediction(2, predict_unchanged())\n"
            "return_result(commit_contrastive(1, selected, [contrast]))"
        )
    )
    result = await GroundingAgent(
        llm=client, workspace=str(tmp_path / "minimal-commitment-surface")
    ).produce(commitment_input())
    assert result["predicted_effect"]["assertions"] == [
        {"row": 0, "col": 1, "value": 7}
    ]


@pytest.mark.asyncio
async def test_predict_cell_values_rejects_list_wrapper_and_unselected_offsets(
    tmp_path: Path,
) -> None:
    value = commitment_input()
    value["accepted_claim"]["primary_region"] = {
        "bbox": {"row": 10, "col": 20, "height": 2, "width": 2},
        "rows": [".#", "##"],
    }
    client = FakeLLMClient()
    client._response_queue.extend(
        [
            response(
                "selected = predict_cell_values([cell_after(d_row=0, d_col=1, value=7)])\n"
                "contrast = action_prediction(2, predict_unchanged())\n"
                "return_result(commit_contrastive(1, selected, [contrast]))"
            ),
            response(
                "selected = predict_cell_values(cell_after(d_row=0, d_col=0, value=7))\n"
                "contrast = action_prediction(2, predict_unchanged())\n"
                "commit_contrastive(1, selected, [contrast])"
            ),
            response(
                "selected = predict_cell_values(cell_after(d_row=0, d_col=1, value=7))\n"
                "contrast = action_prediction(2, predict_unchanged())\n"
                "return_result(commit_contrastive(1, selected, [contrast]))"
            ),
        ]
    )
    result = await GroundingAgent(
        llm=client,
        workspace=str(tmp_path / "relative-cell-boundaries"),
        max_iterations=3,
        max_retries=3,
    ).produce(value)
    assert result["predicted_effect"]["assertions"] == [
        {"row": 10, "col": 21, "value": 7}
    ]
    assert not client._response_queue


@pytest.mark.asyncio
async def test_three_explorations_still_allow_commitment_submission(
    tmp_path: Path,
) -> None:
    client = FakeLLMClient()
    client._response_queue.extend(
        [
            response("inspect_evidence()['current_frame']['grid']"),
            response("len(inspect_evidence()['transitions'])"),
            response("accepted = inspect_evidence()['accepted_claim']"),
            response(
                "selected = predict_cell_values(cell_after(d_row=0, d_col=0, value=7))\n"
                "contrast = action_prediction(2, predict_unchanged())\n"
                "return_result(commit_contrastive(1, selected, [contrast]))"
            ),
        ]
    )
    agent = GroundingAgent(
        llm=client,
        workspace=str(tmp_path / "bounded-exploration"),
        max_iterations=6,
        max_retries=2,
    )
    result = await agent.produce(commitment_input())
    assert result["predicted_effect"] == {
        "kind": "replace",
        "source": "primary_region",
        "expected_after": {
            "bbox": {"row": 0, "col": 1, "height": 1, "width": 1},
            "rows": ["#"],
        },
        "assertions": [{"row": 0, "col": 1, "value": 7}],
    }
    assert result["alternative_predictions"][0]["action"] == 2
    assert not client._response_queue


@pytest.mark.asyncio
async def test_three_explorations_still_allow_qwen_style_assignment_claim(
    tmp_path: Path,
) -> None:
    client = FakeLLMClient()
    client._response_queue.extend(
        [
            response("inspect_evidence()['current_frame']['grid']"),
            response("len(inspect_evidence()['transitions'])"),
            response("len(inspect_evidence()['probe_results'])"),
            response(
                "primary = make_mask(row=0, col=1, rows=['#'])\n"
                "result = submit_grounding_claim(primary_region=primary, alternatives=[])\n"
                "print(result)\n"
                "raise AssertionError('submission must terminate before later statements')"
            ),
        ]
    )
    agent = GroundingAgent(
        llm=client,
        workspace=str(tmp_path / "qwen-constructor-assignment"),
        max_iterations=4,
        max_retries=2,
    )
    result = await agent.produce(artifact_input())
    assert result["primary_region"] == {
        "bbox": {"row": 0, "col": 1, "height": 1, "width": 1},
        "rows": ["#"],
    }
    assert result["alternatives"] == []


@pytest.mark.asyncio
async def test_evidence_read_and_submission_remain_available_after_three_explorations(
    tmp_path: Path,
) -> None:
    client = FakeLLMClient()
    client._response_queue.extend(
        [
            response("inspect_evidence()['current_frame']['grid']"),
            response("len(inspect_evidence()['transitions'])"),
            response("inspect_evidence()['current_probe_result']['change_accounting']"),
            response(
                "state = inspect_evidence()\n"
                "update = account_prior(disposition='revise', "
                "rationale='The prior region did not change; test a different grouping.')\n"
                "h = provisional_heuristic(name='replacement', "
                "belief='Another region may respond to the selected action.', "
                "scope='Current attempt and observed probe.', decision_role='active', "
                "reconsider_when='Its prospective effect mismatches.')\n"
                "primary = make_mask(row=0, col=1, rows=['#'])\n"
                "return_result(submit_grounding_claim("
                "primary_region=primary, alternatives=[], hypothesis_update=update, "
                "heuristic_updates=[h]))"
            ),
        ]
    )
    agent = GroundingAgent(
        llm=client,
        workspace=str(tmp_path / "post-limit-evidence-read"),
        max_iterations=4,
        max_retries=2,
    )
    result = await agent.produce(post_probe_claim_input(with_prior_heuristic=True))
    assert result["primary_region"]["bbox"] == {
        "row": 0,
        "col": 1,
        "height": 1,
        "width": 1,
    }
    assert result["heuristic_updates"][0]["name"] == "replacement"
    assert "evidence_refs" not in result["heuristic_updates"][0]
    assert not client._response_queue


@pytest.mark.asyncio
async def test_artifact_iterations_remain_bounded_without_a_submission(tmp_path: Path) -> None:
    client = FakeLLMClient()
    client._response_queue.extend(
        [
            response("inspect_evidence()['current_frame']['grid']"),
            response("len(inspect_evidence()['transitions'])"),
            response("len(inspect_evidence()['probe_results'])"),
        ]
    )
    agent = GroundingAgent(
        llm=client,
        workspace=str(tmp_path / "bounded-by-codeact"),
        max_iterations=3,
        max_retries=1,
    )
    with pytest.raises(GenerationError):
        await agent.produce(artifact_input())


@pytest.mark.asyncio
async def test_model_visible_make_mask_derives_dimensions(tmp_path: Path) -> None:
    client = FakeLLMClient()
    client._response_queue.append(
        response(
            "rows = ['##.##.#', '##.##.#']\n"
            "candidate = {'observation_id':'frame-a',"
            "'primary_region':make_mask(0, 0, rows),'alternatives':[],"
            "'evidence_refs':['frame-a'],'parent_claim_id':None}\n"
            "return_result(candidate)"
        )
    )
    agent = GroundingAgent(llm=client, workspace=str(tmp_path / "mask-helper"))
    result = await agent.produce(artifact_input())
    assert result["primary_region"]["bbox"] == {
        "row": 0,
        "col": 0,
        "height": 2,
        "width": 7,
    }


@pytest.mark.asyncio
async def test_claim_submitter_inserts_current_evidence_id(tmp_path: Path) -> None:
    client = FakeLLMClient()
    client._response_queue.append(
        response(
            "return_result(submit_grounding_claim("
            "primary_region=make_mask(0,1,['#']), alternatives=[]))"
        )
    )
    result = await GroundingAgent(
        llm=client, workspace=str(tmp_path / "claim-submitter")
    ).produce(artifact_input())
    assert result["evidence_refs"] == ["frame-a"]


@pytest.mark.asyncio
async def test_agent_repairs_live_mask_width_error_within_same_artifact(tmp_path: Path) -> None:
    invalid = {
        "observation_id": "frame-a",
        "primary_region": {
            "bbox": {"row": 0, "col": 0, "height": 1, "width": 8},
            "rows": ["##.##.#"],
        },
        "alternatives": [],
        "evidence_refs": ["frame-a"],
        "parent_claim_id": None,
    }
    corrected = {
        **invalid,
        "primary_region": {
            "bbox": {"row": 0, "col": 0, "height": 1, "width": 7},
            "rows": ["##.##.#"],
        },
    }
    client = FakeLLMClient()
    client._response_queue.extend(
        [
            response(f"candidate = {invalid!r}\nreturn_result(candidate)"),
            response(f"candidate = {corrected!r}\nreturn_result(candidate)"),
        ]
    )
    agent = GroundingAgent(
        llm=client,
        workspace=str(tmp_path / "repair-workspace"),
        max_iterations=12,
        max_retries=4,
    )
    result = await agent.produce(artifact_input())
    assert result["primary_region"] == corrected["primary_region"]


@pytest.mark.asyncio
async def test_required_usage_counts_input_plus_output_and_rejects_missing_fields(tmp_path: Path) -> None:
    code = (
        "return_result({'observation_id':'frame-a',"
        "'primary_region':{'bbox':{'row':0,'col':1,'height':1,'width':1},'rows':['#']},"
        "'alternatives':[{'bbox':{'row':0,'col':0,'height':1,'width':1},'rows':['#']}],"
        "'evidence_refs':['frame-a'],'parent_claim_id':None})"
    )
    good = FakeLLMClient(); good._response_queue.append(response(code, usage={"input_tokens": 9, "output_tokens": 4, "reasoning_tokens": 20}))
    agent = GroundingAgent(llm=good, workspace=str(tmp_path / "good"), require_usage=True)
    await agent.produce(artifact_input())
    assert agent.usage == [
        {
            "input_tokens": 9,
            "output_tokens": 4,
            "reasoning_tokens": 20,
            "cache_read_tokens": None,
            "cache_write_tokens": None,
        }
    ]

    bad = FakeLLMClient(); bad._response_queue.append(response(code, usage={"total_tokens": 13}))
    missing = GroundingAgent(llm=bad, workspace=str(tmp_path / "bad"), require_usage=True, max_retries=1)
    with pytest.raises(Exception) as captured:
        await missing.produce(artifact_input())
    messages = []
    current: BaseException | None = captured.value
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current)); messages.append(str(current)); current = current.__cause__ or current.__context__
    assert any("input_tokens and output_tokens" in message for message in messages)


@pytest.mark.asyncio
async def test_tail_from_start_reads_preexisting_first_line_and_waits_on_partial_line(tmp_path: Path) -> None:
    path = tmp_path / "requests.jsonl"
    path.write_text("first\npart", encoding="utf-8")
    tail = _tail_from_start(str(path), poll_interval=0.01)
    assert await anext(tail) == "first"
    with path.open("a", encoding="utf-8") as handle:
        handle.write("ial\nsecond\n")
    assert await anext(tail) == "partial"
    assert await anext(tail) == "second"
    await tail.aclose()


def test_provider_worker_requires_the_frozen_route_and_explicit_credential(monkeypatch) -> None:
    class Provider:
        def __init__(self) -> None:
            self.config = {}

    provider = Provider()
    captured = {}
    monkeypatch.setenv("ARC_LLM_API_KEY", "test-secret")
    monkeypatch.setattr(launcher, "build_llm", lambda profile, *, api_key: provider)

    def wrap(base, profile, **kwargs):
        captured.update({"base": base, "profile": profile, **kwargs})
        return "wrapped"

    monkeypatch.setattr(launcher, "AuditedProviderClient", wrap)
    request = {
        "worker_mode": "provider",
        "provider_request_profile": {"schema": "fixture"},
        "provider_route": LIVE_PROVIDER_ROUTE,
        "transport_max_retries": 5,
    }
    client, live = _client_for(request)
    assert client == "wrapped" and live is True
    assert provider.config["extra_body"] == {"provider": LIVE_PROVIDER_ROUTE}
    assert captured["known_credentials"] == ("test-secret",)
    with pytest.raises(ValueError, match="frozen provider route"):
        _client_for({**request, "provider_route": {}})


@pytest.mark.asyncio
async def test_http_429_retries_with_exponential_backoff_and_preserves_attempts(monkeypatch) -> None:
    class RateLimit(Exception):
        status_code = 429
        response = None

    expected = object()

    class Provider:
        def __init__(self) -> None:
            self.calls = 0
            self.last_tools = []
            self.last_tool_choice = None

        async def acall(self, *_args, **_kwargs):
            self.last_tools = list(_kwargs.get("tools") or ())
            self.last_tool_choice = _kwargs.get("tool_choice")
            self.calls += 1
            if self.calls < 3:
                raise RateLimit("upstream throttled")
            return expected

    sleeps: list[float] = []

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(provider_module.asyncio, "sleep", sleep)
    monkeypatch.setattr(provider_module.random, "uniform", lambda _low, _high: 0.0)
    wrapped = AuditedProviderClient(
        Provider(),
        {
            "schema": "0002-nooa-provider-request-v2",
            "api_surface": "responses",
            "model": "openai/qwen/qwen3.6-27b",
            "wire_model": "qwen/qwen3.6-27b",
            "api_base": "https://openrouter.ai/api/v1",
            "max_output_tokens": 8192,
            "reasoning_effort": "medium",
            "request_timeout_seconds": 180.0,
            "temperature": None,
            "seed": None,
            "stream": False,
            "drop_params": True,
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "truncation": "disabled",
            "nooa_max_retries": 0,
            "nooa_rate_limit_extra_retries": 0,
            "litellm_num_retries": 0,
        },
        transport_max_retries=5,
    )

    class ExecutePythonTool:
        name = "execute_python"
        description = "execute"

        def get_parameter_schema(self, strict=True):
            assert strict is True
            return {"type": "object"}

    class DirectResultTool(ExecutePythonTool):
        name = "return_result"

    assert await wrapped.acall([], tools=[ExecutePythonTool(), DirectResultTool()]) is expected
    assert [tool.name for tool in wrapped.provider.last_tools] == ["execute_python"]
    assert wrapped.provider.last_tool_choice == "auto"
    assert sleeps == [2.0, 4.0]
    attempts = wrapped.audit_calls[0]["attempts"]
    assert [item["outcome"] for item in attempts] == [
        "retryable_failure",
        "retryable_failure",
        "success",
    ]
    assert [item.get("retry_delay_source") for item in attempts] == [
        "exponential-jitter",
        "exponential-jitter",
        None,
    ]
