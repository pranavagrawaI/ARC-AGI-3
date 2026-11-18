#!/usr/bin/env python3
"""Produce a persistent scripted control/treatment trace and score it offline.

Runs real submission and host-evidence methods. No model or ARC environment;
these artifacts are explicitly ineligible for an experimental arm PASS.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from test_q6_contract import Session, chassis, scorer  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    scores = []
    for arm in ("control", "treatment"):
        run = args.output / arm
        session = Session(run, treatment=arm == "treatment")
        manifest_path = (
            Path(__file__).resolve().parents[3]
            / "deployment/kaggle/q6-nooa-candidate/CHASSIS.json"
        )
        manifest = json.loads(manifest_path.read_text())
        launch_path = run / "launch.json"
        launch = json.loads(launch_path.read_text())
        launch.update(
            manifest=manifest,
            contract_sha256=manifest["contract_sha256"],
            run_id=uuid.uuid4().hex,
            session_id=uuid.uuid4().hex,
        )
        launch_path.write_text(json.dumps(launch, indent=2) + "\n")
        session.seed()
        if arm == "control":
            assert session.submit(["UP"]) is None
            session.execute()
        else:
            assert session.submit(["UP"]).startswith("REJECTED:")
            try:
                session.actor.resolve_stall(
                    "continue",
                    ["old"],
                    ["old"],
                    "DOWN",
                    "test the governing prediction",
                    abandon_if="no_impact",
                )
            except chassis._ReturnResultSignal:
                pass
            else:
                raise AssertionError("resolution did not commit its action")
            session.execute(inert=True)
            session.actor.finish_stall(
                ["revised"], "the observation falsified the old basis"
            )
        for rationale in (
            "alpha beta gamma",
            "delta epsilon zeta",
            "eta theta iota",
            "kappa lambda mu",
        ):
            assert session.submit(["UP"] * 5, rationale=rationale) is None
            session.execute()
        score = scorer.score_game(run)
        assert score["status"] == "SCORED", score
        assert score["P9"] == 1
        scores.append(score)
    report = {"runs": scores, "arm": scorer.score_arm(scores)}
    assert report["arm"]["status"] != "PASS"
    (args.output / "SCORE.json").write_text(json.dumps(report, indent=2) + "\n")
    print(args.output / "SCORE.json")


if __name__ == "__main__":
    main()
