from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from host.evidence import EvidenceStore
from host.ipc import (
    ArtifactBridge,
    ArtifactGenerationError,
    ArtifactWorkerError,
    select_exact_response,
    validate_response_record,
)
from host.kernel import GroundingKernel, QualificationPaused
from host.protocol import grounding_claim_from_draft

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "nooa-runtime"


def _mask(row: int, col: int, height: int = 1, width: int = 1) -> dict:
    return {"bbox": {"row": row, "col": col, "height": height, "width": width}, "rows": ["#" * width] * height}


def _contrastive_translate(
    action: int,
    d_row: int,
    d_col: int,
    *,
    alternative_action: int,
    alternative_d_row: int,
    alternative_d_col: int,
) -> str:
    return (
        f"selected = predict_translate({d_row}, {d_col})\n"
        f"alternative = action_prediction({alternative_action}, "
        f"predict_translate({alternative_d_row}, {alternative_d_col}))\n"
        f"return_result(commit_contrastive({action}, selected, [alternative]))"
    )


def test_actual_nooa_sandbox_surface_is_minimal_and_fresh_per_artifact(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path / "evidence")
    frame = store.record_frame([[0, 7], [0, 0]], level=0, turn=0, attempt_id="a", available_actions=[1, 2])

    def fixture(phase, artifact_input):
        if phase == "claim":
            return [
                (
                    "assert inspect_evidence()['visible_frame_window'][0]['evidence_id'] == inspect_evidence()['current_frame']['evidence_id']\n"
                    "assert inspect_evidence()['transitions'] == () and inspect_evidence()['attempts'][0]['attempt_id'] == 'a'\n"
                    "try:\n    self\nexcept NameError:\n    pass\nelse:\n    raise AssertionError('self leaked')\n"
                    "leak = 99\n"
                    "return_result(submit_grounding_claim("
                    "primary_region=make_mask(0,1,['#']), alternatives=[make_mask(1,1,['#'])]))"
                )
            ]
        return [
            (
                "try:\n    leak\nexcept NameError:\n    pass\nelse:\n    raise AssertionError('artifact state leaked')\n"
                "try:\n    world_model\nexcept NameError:\n    pass\nelse:\n    raise AssertionError('advanced capability leaked')\n"
                "try:\n    make_mask\nexcept NameError:\n    pass\nelse:\n    raise AssertionError('claim constructor leaked')\n"
                "selected = predict_cell_values(cell_after(d_row=0, d_col=0, value=7))\n"
                "alternative = action_prediction(2, predict_unchanged())\n"
                "return_result(commit_contrastive(1, selected, [alternative]))"
            )
        ]

    bridge = ArtifactBridge(runtime_root=RUNTIME, run_root=tmp_path / "run", fixture_policy=fixture)
    claim_draft = bridge.request(
        phase="claim", observation_id=frame["evidence_id"], evidence=store.model_view(), legal_actions=[1, 2]
    )
    claim_value = grounding_claim_from_draft(claim_draft)
    commitment = bridge.request(
        phase="commitment",
        observation_id=frame["evidence_id"],
        evidence=store.model_view(),
        legal_actions=[1, 2],
        accepted_claim=claim_value.to_record(),
    )
    assert commitment["claim_id"] == claim_value.claim_id
    assert commitment["predicted_effect"]["assertions"] == [
        {"row": 0, "col": 1, "value": 7}
    ]
    assert list((tmp_path / "run" / "artifact-workspaces").iterdir()) == []
    retired = [json.loads(line) for line in (tmp_path / "run/trace.jsonl").read_text().splitlines() if "workspace_retired" in line]
    assert len(retired) == 2 and all(len(item["workspace_sha256"]) == 64 for item in retired)
    visible = json.dumps(bridge.latest_response["audit"]["model_calls"]).lower()
    assert all(term not in visible for term in ("ls20", "the player", "goal identity", "blocked terrain", "passable terrain"))
    assert all(secret not in visible for secret in ("arc_api_key", "openrouter_api_key"))


def test_actual_sandbox_blocks_host_imports_and_preserves_repair_feedback(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path / "evidence")
    frame = store.record_frame([[0, 7]], level=0, turn=0, attempt_id="a", available_actions=[1, 2])
    valid = (
        "return_result({'observation_id': inspect_evidence()['current_frame']['evidence_id'],"
        "'primary_region':{'bbox':{'row':0,'col':1,'height':1,'width':1},'rows':['#']},"
        "'alternatives':[{'bbox':{'row':0,'col':0,'height':1,'width':1},'rows':['#']}],"
        "'evidence_refs':[inspect_evidence()['current_frame']['evidence_id']],'parent_claim_id':None})"
    )
    bridge = ArtifactBridge(
        runtime_root=RUNTIME,
        run_root=tmp_path / "run",
        fixture_policy=lambda _phase, _input: ["import os", valid],
    )
    bridge.request(phase="claim", observation_id=frame["evidence_id"], evidence=store.model_view(), legal_actions=[1, 2])
    calls = bridge.latest_response["audit"]["model_calls"]
    assert len(calls) == 2
    repair_context = json.dumps(calls[1]["request"]["messages"])
    assert "RestrictedCodeError" in repair_context and "os" in repair_context
    assert "/home/hipra" not in repair_context


def test_ipc_rejects_stale_hashes_errors_and_duplicates_without_fallback() -> None:
    request = {
        "protocol": "0158-grounding-ipc-v1", "run_id": "r", "artifact_id": "a",
        "sequence": 0, "phase": "claim", "observation_id": "o", "evidence_sha256": "e",
        "model_contract_sha256": "m", "request_sha256": "q",
    }
    artifact = {"x": 1}
    from host.canonical import content_sha256
    response = {**request, "artifact": artifact, "artifact_sha256": content_sha256(artifact), "complete": True}
    request["request_sha256"] = content_sha256({key: value for key, value in request.items() if key != "request_sha256"})
    response["request_sha256"] = request["request_sha256"]
    assert validate_response_record(request, response) == artifact
    with pytest.raises(ArtifactWorkerError, match="mismatched"):
        validate_response_record(request, {**response, "observation_id": "future"})
    with pytest.raises(ArtifactWorkerError, match="worker error"):
        validate_response_record(request, {**request, "error": {"type": "X", "message": "bad"}, "complete": True})
    with pytest.raises(ArtifactGenerationError, match="GenerationError"):
        validate_response_record(
            request,
            {**request, "error": {"type": "GenerationError", "message": "incomplete"}, "complete": True},
        )
    assert select_exact_response([response], "a") == response
    with pytest.raises(ArtifactWorkerError, match="exactly one"):
        select_exact_response([response, response], "a")


def test_sanitized_historical_shadow_reaches_exact_g2_without_live_contact(tmp_path: Path) -> None:
    fixture_data = json.loads((ROOT / "fixtures/historical-shadow/ls20-public-crop.json").read_text())
    frame_records = fixture_data["frames"]
    masks = ((10, 15), (5, 15), (10, 15))
    effects = ((-5, 0), (5, 0), (0, -5))
    action_effects = {1: (-5, 0), 2: (5, 0), 3: (0, -5)}
    call = {"claim": 0, "commitment": 0}

    def fixture(phase, artifact_input):
        index = call[phase]
        call[phase] += 1
        if phase == "claim":
            row, col = masks[index]
            update = (
                "None"
                if index == 0
                else "account_prior(disposition='retain', "
                "rationale='The tracked block heuristic remains provisional.')"
            )
            return [
                (
                    "h = provisional_heuristic(name='tracked_motion', "
                    "belief='The tracked pattern may move according to the chosen action.', "
                    "scope='Sanitized public shadow transitions only.', "
                    "decision_role='active', reconsider_when='A prospective translation mismatches.')\n"
                    f"primary = make_mask({row}, {col}, {['#####'] * 5!r})\n"
                    "alternative = make_mask(0, 0, ['#'])\n"
                    f"update = {update}\n"
                    "return_result(submit_grounding_claim(primary, [alternative], "
                    "hypothesis_update=update, heuristic_updates=[h]))"
                )
            ]
        d_row, d_col = effects[index]
        action = fixture_data["actions"][index]
        alternatives = [
            (
                "action_prediction("
                f"{alternative_action}, predict_translate({alternative_effect[0]}, "
                f"{alternative_effect[1]}))"
            )
            for alternative_action, alternative_effect in action_effects.items()
            if alternative_action != action
        ]
        return [
            (
                f"selected = predict_translate({d_row}, {d_col})\n"
                f"return_result(commit_contrastive({action}, selected, "
                f"[{', '.join(alternatives)}]))"
            )
        ]

    store = EvidenceStore(tmp_path / "evidence")
    bridge = ArtifactBridge(runtime_root=RUNTIME, run_root=tmp_path / "bridge", fixture_policy=fixture)
    kernel = GroundingKernel(run_root=tmp_path / "run", evidence=store, bridge=bridge)
    kernel.start(frame_records[0]["grid"], attempt_id="shadow-attempt", available_actions=fixture_data["available_actions"])
    for index in range(3):
        kernel.analyze()
        kernel.commit()
        envelope = kernel.issue_one_action()
        assert envelope.action == fixture_data["actions"][index]
        with pytest.raises(RuntimeError, match="already issued"):
            kernel.issue_one_action()
        verdict = kernel.observe_successor(
            frame_records[index + 1]["grid"], available_actions=fixture_data["available_actions"]
        )
        assert verdict.status == "supported"
        if index == 0:
            assert verdict.action_specificity == "inconclusive"
        else:
            assert verdict.action_specificity == "supported" and verdict.promotable
    assert kernel.gates.gate == "G2"
    assert kernel.gates.transition_capability_eligible
    stop = json.loads((tmp_path / "run/review-stops/g2-transition-eligibility.json").read_text())
    assert stop["new_api_exposed"] is False
    assert not (tmp_path / "run/failure-packets").exists() or not any((tmp_path / "run/failure-packets").iterdir())


def test_first_contradicted_successor_atomically_pauses_with_a_failure_packet(tmp_path: Path) -> None:
    def fixture(phase, artifact_input):
        if phase == "claim":
            return [
                (
                    "return_result({'observation_id':inspect_evidence()['current_frame']['evidence_id'],"
                    "'primary_region':{'bbox':{'row':0,'col':0,'height':1,'width':1},'rows':['#']},"
                    "'alternatives':[{'bbox':{'row':0,'col':1,'height':1,'width':1},'rows':['#']}],"
                    "'evidence_refs':[inspect_evidence()['current_frame']['evidence_id']],'parent_claim_id':None})"
                )
            ]
        return [
            _contrastive_translate(
                1, 0, 1, alternative_action=2, alternative_d_row=0, alternative_d_col=-1
            )
        ]

    store = EvidenceStore(tmp_path / "evidence")
    bridge = ArtifactBridge(runtime_root=RUNTIME, run_root=tmp_path / "bridge", fixture_policy=fixture)
    kernel = GroundingKernel(run_root=tmp_path / "run", evidence=store, bridge=bridge)
    kernel.start([[7, 0]], attempt_id="a", available_actions=[1, 2])
    kernel.analyze(); kernel.commit(); kernel.issue_one_action()
    verdict = kernel.observe_successor([[7, 0]], available_actions=[1, 2])
    assert verdict.status == "contradicted"
    assert kernel.gates.gate == "INVALIDATED" and kernel.gates.paused_for_review
    packets = list((tmp_path / "run/failure-packets").iterdir())
    assert len(packets) == 1
    assert json.loads((packets[0] / "review.yaml").read_text())["status"] == "pending"
    assert kernel.pending_commitment is None


def test_autonomous_contradiction_records_probe_and_opens_fresh_grounding_cycle(tmp_path: Path) -> None:
    claim_calls = 0

    def fixture(phase, artifact_input):
        nonlocal claim_calls
        if phase == "claim":
            expected_results = claim_calls
            claim_calls += 1
            update = (
                ""
                if expected_results == 0
                else ", hypothesis_update=account_prior(disposition='revise', "
                "rationale='The prior prediction mismatched this transition.')"
            )
            return [
                (
                    f"assert len(inspect_evidence()['probe_results']) == {expected_results}\n"
                    "h = provisional_heuristic(name='candidate_motion', "
                    "belief='The selected cell may respond to action 1.', "
                    "scope='This attempt and observed transitions.', "
                    "decision_role='active', reconsider_when='The next prospective effect mismatches.')\n"
                    "return_result(submit_grounding_claim("
                    f"primary_region=make_mask(0,0,['#']), alternatives=[]{update}, "
                    "heuristic_updates=[h]))"
                )
            ]
        return [
            _contrastive_translate(
                1, 0, 1, alternative_action=2, alternative_d_row=0, alternative_d_col=-1
            )
        ]

    store = EvidenceStore(tmp_path / "evidence")
    bridge = ArtifactBridge(runtime_root=RUNTIME, run_root=tmp_path / "bridge", fixture_policy=fixture)
    kernel = GroundingKernel(
        run_root=tmp_path / "run", evidence=store, bridge=bridge,
        autonomous_continuation=True, model_frame_window=3,
    )
    kernel.start([[7, 0]], attempt_id="a", available_actions=[1, 2])
    for expected_results in (1, 2):
        kernel.analyze(); kernel.commit(); kernel.issue_one_action()
        verdict = kernel.observe_successor([[7, 0]], available_actions=[1, 2])
        assert verdict.status == "contradicted"
        assert kernel.gates.gate == "EMPTY" and kernel.gates.paused_for_review is False
        assert kernel.current_claim is None and kernel.pending_commitment is None
        assert len(store.probe_results()) == expected_results
    reset_lines = (tmp_path / "run/autonomous-cycle-resets.jsonl").read_text().splitlines()
    reset = json.loads(reset_lines[0])
    assert reset["human_mediation"] is False and reset["new_api_exposed"] is False
    assert len(reset_lines) == 2
    assert not any((tmp_path / "run/failure-packets").iterdir())


def test_contradicted_local_proxy_cannot_drop_visible_change_from_next_claim(tmp_path: Path) -> None:
    claim_calls = 0

    def fixture(phase, artifact_input):
        nonlocal claim_calls
        if phase == "claim":
            claim_calls += 1
            if claim_calls == 1:
                return [
                    (
                        "return_result(submit_grounding_claim("
                        "primary_region=make_mask(0,0,['#']), alternatives=[]))"
                    )
                ]
            return [
                (
                    "prior = inspect_evidence()['probe_results'][-1]\n"
                    "assert prior['change_accounting']['changed_cells_inside_primary'] == 0\n"
                    "assert prior['change_accounting']['changed_cells_outside_primary'] == 1\n"
                    "update = account_prior(disposition='abandon', "
                    "rationale='The local proxy stayed fixed while other visible change remained.')\n"
                    "h = provisional_heuristic(name='candidate_motion', "
                    "belief='The newly selected cell may track the action-varying change.', "
                    "scope='One contradicted proxy transition.', "
                    "decision_role='active', reconsider_when='Its prospective effect mismatches.')\n"
                    "return_result(submit_grounding_claim("
                    "primary_region=make_mask(1,1,['#']), alternatives=[], "
                    "hypothesis_update=update, heuristic_updates=[h]))"
                )
            ]
        return [
            _contrastive_translate(
                1, 0, 1, alternative_action=2, alternative_d_row=1, alternative_d_col=0
            )
        ]

    store = EvidenceStore(tmp_path / "evidence")
    bridge = ArtifactBridge(runtime_root=RUNTIME, run_root=tmp_path / "bridge", fixture_policy=fixture)
    kernel = GroundingKernel(
        run_root=tmp_path / "run", evidence=store, bridge=bridge,
        autonomous_continuation=True,
    )
    kernel.start([[7, 0], [0, 0]], attempt_id="a", available_actions=[1, 2])
    kernel.analyze(); kernel.commit(); kernel.issue_one_action()
    verdict = kernel.observe_successor([[7, 0], [0, 9]], available_actions=[1, 2])
    assert verdict.status == "contradicted"
    kernel.analyze()
    assert kernel.current_claim is not None
    assert kernel.current_claim.primary_region.bbox.row == 1
    assert kernel.current_claim.hypothesis_update is not None
    assert kernel.current_claim.hypothesis_update.disposition == "abandon"
    assert kernel.current_claim.heuristic_register[0].epistemic_status == "provisional"


def test_generation_failure_uses_the_same_packet_shape_and_is_not_substrate(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path / "evidence")
    bridge = ArtifactBridge(
        runtime_root=RUNTIME,
        run_root=tmp_path / "bridge",
        fixture_policy=lambda _phase, _input: ["import os"],
    )
    kernel = GroundingKernel(run_root=tmp_path / "run", evidence=store, bridge=bridge)
    kernel.start([[7, 0]], attempt_id="a", available_actions=[1, 2])
    with pytest.raises(QualificationPaused) as captured:
        kernel.analyze()
    assert captured.value.failure_class == "model_generation"
    causal = json.loads((captured.value.packet / "causal-files.json").read_text())
    assert causal["failure_class"] == "model_generation"
    assert "output budget" in causal["live_causal_hypotheses"][0]
    assert (captured.value.packet / "before.png").is_file()
    assert (captured.value.packet / "after.png").is_file()


def test_environment_failure_packet_uses_commitment_surface_and_preserves_external_phase(tmp_path: Path) -> None:
    def fixture(phase, artifact_input):
        if phase == "claim":
            return ["return_result(submit_grounding_claim(make_mask(0,0,['#']), []))"]
        return [
            _contrastive_translate(
                1, 0, 1, alternative_action=2, alternative_d_row=0, alternative_d_col=-1
            )
        ]

    store = EvidenceStore(tmp_path / "evidence")
    bridge = ArtifactBridge(runtime_root=RUNTIME, run_root=tmp_path / "bridge", fixture_policy=fixture)
    kernel = GroundingKernel(run_root=tmp_path / "run", evidence=store, bridge=bridge)
    kernel.start([[7, 0]], attempt_id="a", available_actions=[1, 2])
    kernel.analyze(); kernel.commit(); kernel.issue_one_action()
    with pytest.raises(QualificationPaused) as captured:
        kernel.pause_external_failure(RuntimeError("no successor"), phase="environment_step")
    surface = json.loads((captured.value.packet / "callable-surface.json").read_text())
    assert surface["external_failure_phase"] == "environment_step"
    assert "inspect_evidence" in surface["names"]
    assert not {
        "artifact_input",
        "frames",
        "current_frame",
        "frame_summaries",
        "transitions",
        "probe_results",
        "attempts",
    } & set(surface["names"])
    assert "commit_contrastive" in surface["names"]
    assert "predict_cell_values" in surface["names"]
    assert "make_mask" not in surface["names"]
    assert "predict_replace" not in surface["names"]
    assert "submit_probe" not in surface["names"]


def test_worker_timeout_kills_the_process_group_flushes_trace_and_retires_scratch(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path / "evidence")
    frame = store.record_frame([[7]], level=0, turn=0, attempt_id="a", available_actions=[1, 2])
    bridge = ArtifactBridge(
        runtime_root=RUNTIME,
        run_root=tmp_path / "bridge",
        fixture_policy=lambda _phase, _input: ["while True:\n    pass"],
        timeout_seconds=0.5,
    )
    with pytest.raises(ArtifactWorkerError, match="timed out"):
        bridge.request(phase="claim", observation_id=frame["evidence_id"], evidence=store.model_view(), legal_actions=[1, 2])
    trace = [json.loads(line) for line in (tmp_path / "bridge/trace.jsonl").read_text().splitlines()]
    pid = next(item["pid"] for item in trace if item["kind"] == "artifact_worker_started")
    assert any(item["kind"] == "artifact_timeout" for item in trace)
    assert any(item["kind"] == "artifact_workspace_retired" for item in trace)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    scratch_root = tmp_path / "bridge/artifact-workspaces"
    assert not scratch_root.exists() or list(scratch_root.iterdir()) == []


def test_terminal_and_death_successors_never_enter_ordinary_transitions(tmp_path: Path) -> None:
    def fixture(phase, artifact_input):
        if phase == "claim":
            return [
                (
                    "return_result({'observation_id':inspect_evidence()['current_frame']['evidence_id'],"
                    "'primary_region':{'bbox':{'row':0,'col':0,'height':1,'width':1},'rows':['#']},"
                    "'alternatives':[{'bbox':{'row':0,'col':1,'height':1,'width':1},'rows':['#']}],"
                    "'evidence_refs':[inspect_evidence()['current_frame']['evidence_id']],'parent_claim_id':None})"
                )
            ]
        return [
            _contrastive_translate(
                1, 0, 1, alternative_action=2, alternative_d_row=0, alternative_d_col=-1
            )
        ]

    store = EvidenceStore(tmp_path / "terminal-evidence")
    bridge = ArtifactBridge(runtime_root=RUNTIME, run_root=tmp_path / "terminal-bridge", fixture_policy=fixture)
    kernel = GroundingKernel(run_root=tmp_path / "terminal-run", evidence=store, bridge=bridge)
    kernel.start([[7, 0]], attempt_id="a", available_actions=[1, 2])
    kernel.analyze(); kernel.commit(); kernel.issue_one_action()
    event = kernel.observe_terminal_successor([[0, 7]])
    assert event["kind"] == "terminal"
    assert store.transitions() == [] and len(store.events()["terminal"]) == 1
    kernel.begin_new_level([[5, 0]], level=1, attempt_id="b", available_actions=[1, 2])
    assert kernel.gates.gate == "EMPTY"

    death_store = EvidenceStore(tmp_path / "death-evidence")
    death_bridge = ArtifactBridge(runtime_root=RUNTIME, run_root=tmp_path / "death-bridge", fixture_policy=fixture)
    death_kernel = GroundingKernel(run_root=tmp_path / "death-run", evidence=death_store, bridge=death_bridge)
    death_kernel.start([[7, 0]], attempt_id="a", available_actions=[1, 2])
    death_kernel.analyze(); death_kernel.commit(); death_kernel.issue_one_action()
    with pytest.raises(QualificationPaused):
        death_kernel.observe_death_successor([[9, 9]])
    assert death_store.transitions() == [] and len(death_store.events()["death"]) == 1
