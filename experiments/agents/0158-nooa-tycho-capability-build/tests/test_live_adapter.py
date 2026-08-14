from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
import run_live_ls20
from host.ipc import REVIEW_STOP_POLICY
from host.ipc import ArtifactBridge as RealArtifactBridge

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "nooa-runtime"


def test_environment_timeout_extension_changes_timeout_without_retrying() -> None:
    calls = []

    class Session:
        def post(self, url, *args, **kwargs):
            calls.append((url, args, kwargs))
            return "response"

    environment = SimpleNamespace(_session=Session())
    run_live_ls20._extend_environment_read_timeout(environment, seconds=60.0)
    assert environment._session.post("url", json={}, timeout=10) == "response"
    assert calls == [("url", (), {"json": {}, "timeout": 60.0})]


def _mask(row: int, col: int, height: int = 1, width: int = 1) -> dict:
    return {
        "bbox": {"row": row, "col": col, "height": height, "width": width},
        "rows": ["#" * width] * height,
    }


def test_live_adapter_reaches_terminal_g2_offline_without_provider_or_arc_contact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture_data = json.loads((ROOT / "fixtures/historical-shadow/ls20-public-crop.json").read_text())
    masks = ((10, 15), (5, 15), (10, 15))
    effects = ((-5, 0), (5, 0), (0, -5))
    action_effects = dict(zip(fixture_data["actions"], effects, strict=True))
    calls = {"claim": 0, "commitment": 0}

    def fixture(phase, artifact_input):
        index = calls[phase]
        calls[phase] += 1
        if phase == "claim":
            row, col = masks[index]
            update = (
                "None"
                if index == 0
                else "account_prior(disposition='retain', "
                "rationale='The tracked motion heuristic remains provisional.')"
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

    def bridge_factory(**kwargs):
        return RealArtifactBridge(
            runtime_root=RUNTIME,
            run_root=kwargs["run_root"],
            fixture_policy=fixture,
            timeout_seconds=kwargs["timeout_seconds"],
        )

    class FakeEnvironment:
        def __init__(self) -> None:
            self.index = 0

        def _frame(self):
            source = fixture_data["frames"][self.index]
            return SimpleNamespace(
                frame=[source["grid"]],
                available_actions=fixture_data["available_actions"],
                levels_completed=0,
                state=SimpleNamespace(value="NOT_FINISHED"),
                full_reset=False,
            )

        def reset(self):
            return self._frame()

        def step(self, action, *, data, reasoning):
            assert int(action.value) == fixture_data["actions"][self.index]
            assert data is None and set(reasoning) == {"claim_id", "commitment_id"}
            self.index += 1
            return self._frame()

        def close(self) -> None:
            pass

    class FakeArcade:
        def __init__(self, **_kwargs) -> None:
            pass

        def make(self, game, *, seed, save_recording):
            assert (game, seed, save_recording) == ("ls20", 0, True)
            return FakeEnvironment()

    monkeypatch.setattr(run_live_ls20, "ArtifactBridge", bridge_factory)
    monkeypatch.setattr(run_live_ls20, "RUN_POLICY", REVIEW_STOP_POLICY)
    monkeypatch.setattr(run_live_ls20, "AUTONOMOUS_CONTINUATION", False)
    monkeypatch.setattr(run_live_ls20, "MODEL_FRAME_WINDOW", None)
    monkeypatch.setattr(run_live_ls20.arc_agi, "Arcade", FakeArcade)
    approval = json.loads(run_live_ls20.APPROVAL.read_text(encoding="utf-8"))
    approval["approved_codeact_errors"] = run_live_ls20.MAX_CODEACT_ERRORS
    approval["approved_evidence_access_policy"] = run_live_ls20.EVIDENCE_ACCESS_POLICY
    approval["approved_commitment_surface"] = run_live_ls20.COMMITMENT_SURFACE
    approval["approved_submission_grammar"] = run_live_ls20.SUBMISSION_GRAMMAR
    approval["approved_model_call_cap"] = run_live_ls20.MAX_MODEL_CALLS
    approval["approved_reported_token_cap"] = run_live_ls20.MAX_REPORTED_TOKENS
    approval["approved_active_seconds_cap"] = run_live_ls20.MAX_ACTIVE_SECONDS
    approval["approved_run_policy"] = REVIEW_STOP_POLICY["mode"]
    approval["approved_model_frame_window"] = None
    approval["approved_environment_read_timeout_seconds"] = (
        run_live_ls20.ENVIRONMENT_READ_TIMEOUT_SECONDS
    )
    approval["approved_environment_post_retries"] = 0
    approval["qualification_summary_sha256"] = sha256(
        run_live_ls20.QUALIFICATION.read_bytes()
    ).hexdigest()
    approval["approved_candidate_manifest_sha256"] = run_live_ls20._candidate_manifest()[
        "manifest_sha256"
    ]
    offline_approval = tmp_path / "approval.json"
    offline_approval.write_text(json.dumps(approval), encoding="utf-8")
    monkeypatch.setattr(run_live_ls20, "APPROVAL", offline_approval)
    monkeypatch.setattr(run_live_ls20, "CONSUMPTION", tmp_path / "approval-consumed.json")
    monkeypatch.setenv("ARC_API_KEY", "offline-arc-key")
    monkeypatch.setenv("ARC_LLM_API_KEY", "offline-llm-key")
    summary = run_live_ls20.run(tmp_path / "live")
    assert summary["actions"] == 3
    assert summary["levels_completed"] == 0
    assert summary["stop_reason"] == "g2_transition_eligibility"
    assert summary["scorecard"]["resumable"] is False
    assert summary["scorecard"]["state"] == "STOPPED_AT_G2"
    assert (tmp_path / "approval-consumed.json").is_file()
    with pytest.raises(PermissionError, match="already been consumed"):
        run_live_ls20.run(tmp_path / "second-live")
