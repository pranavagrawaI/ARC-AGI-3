"""Offline release-gate regressions; fabricated fixture records authorize no run."""

import json
from pathlib import Path

import pytest

import run_pair
from transport import Stop


def records(
    tmp_path, task="operational-theory-T1R6", decision="AUTHORIZE_REPLACEMENT_PAIR"
):
    review = tmp_path / "fixture-review.json"
    auth = tmp_path / "fixture-auth.json"
    review.write_text(
        json.dumps(
            dict(
                role="independent-reviewer",
                decision="PASS",
                task=task,
                manifest_sha256="fixture",
            )
        )
    )
    auth.write_text(
        json.dumps(
            dict(
                role="user",
                decision=decision,
                task=task,
                manifest_sha256="fixture",
                metadata_sha256="metadata-fixture",
            )
        )
    )
    return review, auth


@pytest.mark.parametrize("old", [True, False])
def test_release_requires_fresh_identity_and_authority(tmp_path, monkeypatch, old):
    monkeypatch.setattr(run_pair, "verify", lambda: "fixture")
    monkeypatch.setattr(run_pair, "metadata_check", lambda path: "metadata-fixture")
    args = (
        records(tmp_path, "operational-theory-T1R3", "AUTHORIZE_T3")
        if old
        else records(tmp_path)
    )
    if old:
        with pytest.raises(Stop, match="T6 review"):
            run_pair.gates(*args, Path("fixture"))
    else:
        assert run_pair.gates(*args, Path("fixture")) == ("fixture", "metadata-fixture")


def test_candidate_claim_blocks_second_output(tmp_path, monkeypatch):
    monkeypatch.setattr(run_pair, "ROOT", tmp_path)
    (tmp_path / "analysis").mkdir()
    review, auth = records(tmp_path)
    first = tmp_path / "analysis/operational-theory-replacement-20260918T000000Z"
    second = tmp_path / "analysis/operational-theory-replacement-20260918T010000Z"
    live = run_pair.claim_output(first, "fixture", "metadata-fixture", review, auth)
    assert (
        json.loads((live / "CONSUMED.json").read_text())["manifest_sha256"] == "fixture"
    )
    with pytest.raises(FileExistsError):
        run_pair.claim_output(second, "fixture", "metadata-fixture", review, auth)
    assert not second.exists()


@pytest.mark.parametrize(
    "location",
    [
        "candidate/live",
        "analysis/operational-theory-t3-20260917T134947Z",
        "analysis/arbitrary",
    ],
)
def test_output_must_follow_external_convention(tmp_path, monkeypatch, location):
    monkeypatch.setattr(run_pair, "ROOT", tmp_path)
    review, auth = records(tmp_path)
    with pytest.raises(Stop, match="fresh analysis"):
        run_pair.claim_output(
            tmp_path / location, "fixture", "metadata-fixture", review, auth
        )
    assert not (tmp_path / "analysis/operational-theory-replacement-claims").exists()
