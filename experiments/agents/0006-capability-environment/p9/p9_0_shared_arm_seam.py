"""P9.0 — one factory, two arms, one authorized difference.

Question: can 0002-R and a provisional 0006 treatment be constructed from one
factory such that a normalized diff shows only the declared treatment surface
differing, and does that diff actually catch an unauthorized difference?

Two comparisons run. The **declared** comparison diffs capability manifests read
off the constructed arms. The **observed** comparison drives both arms through
an identical scripted fixture over the same recorded observation and diffs the
exact request payloads each arm sent. A negative control then injects an
unauthorized difference and requires the verifier to reject it.

Run: ``uv run python3 experiments/agents/0006-capability-environment/p9/p9_0_shared_arm_seam.py``
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

if __package__ in {None, ""}:  # direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "p9"

from p9.arm_diff import (
    UnauthorizedArmDifference,
    assert_authorized,
    compare_arms,
    normalize_request_payload,
)
from p9.arms import (
    Arm,
    ArmSpec,
    arm_manifest,
    build_arm,
    control_spec,
    participating_sources,
    treatment_spec,
)
from p9.capabilities import authorized_manifest_paths, authorized_request_payload_paths
from p9.evidence import Checks, artifacts_root
from p9.fixtures import (
    RecordedEnvironment,
    ScriptedChatClient,
    action_turn,
    default_recording,
    load_recorded_frames,
    tool_turn,
)

from experiments.harness import run_episode


# Pinned substrate, not an arm-vs-arm comparison. The diff verifier catches
# divergence between arms; a surface added to *both* arms is invisible to it, so
# the shared episode contract is asserted against a literal here.
FROZEN_OBSERVATION_KEYS = [
    "current_metadata",
    "current_observation_file",
    "deliberation",
    "environment_observation",
    "full_history_file",
    "notes",
    "observations_glob",
    "recent_transitions",
    "representations",
    "step",
]
# Top level keeps only what is not already inside environment_observation.
FROZEN_TOP_LEVEL_KEYS = ["deliberation", "environment_observation", "notes", "step"]


PROBE_SCRIPT = (
    tool_turn("import json; print(sorted(json.load(open('current_observation.json'))))"),
    action_turn(1, reason="fixture probe"),
)


def _drive(arm: Arm, frames) -> None:
    run_episode(RecordedEnvironment(frames=frames), arm.agent, max_actions=1)


def main() -> int:
    checks = Checks(
        prototype="P9.0-shared-arm-seam",
        question=(
            "Do 0002-R and a provisional treatment built from one factory differ "
            "only on the declared sentinel capability, and does the verifier catch "
            "an undeclared difference?"
        ),
    )
    repo = Path(__file__).resolve().parents[4]
    frames = load_recorded_frames(default_recording(repo, minimum_frames=2), count=2, crop=8)

    control = control_spec(max_actions=1, max_model_calls=8)
    treatment = treatment_spec(control)

    shared_fields = {
        name
        for name in ArmSpec.__dataclass_fields__
        if name not in {"arm_id", "treatment_capabilities"}
    }
    checks.record(
        "treatment_derives_every_shared_field_from_control",
        all(getattr(control, name) == getattr(treatment, name) for name in shared_fields),
        {"shared_fields": sorted(shared_fields)},
    )

    with TemporaryDirectory(prefix="p9-0-") as scratch:
        root = Path(scratch)
        control_client = ScriptedChatClient(PROBE_SCRIPT)
        treatment_client = ScriptedChatClient(PROBE_SCRIPT)
        with build_arm(control, run_dir=root / "control", inner_client=control_client) as left, \
             build_arm(treatment, run_dir=root / "treatment", inner_client=treatment_client) as right:
            control_manifest = arm_manifest(left)
            treatment_manifest = arm_manifest(right)
            authorized = authorized_manifest_paths(treatment.treatment_capabilities)

            declared = compare_arms(
                control_manifest, treatment_manifest, authorized_paths=authorized
            )
            checks.record(
                "declared_diff_has_no_unauthorized_paths",
                declared.ok,
                declared.as_json(),
            )
            checks.record(
                "declared_diff_exercises_every_authorized_path",
                not declared.authorized_unused,
                {"unused": list(declared.authorized_unused)},
            )
            checks.record(
                "declared_diff_shares_most_of_the_surface",
                declared.shared_paths > 0
                and declared.shared_paths > len(declared.authorized_used),
                {
                    "shared_paths": declared.shared_paths,
                    "compared_paths": declared.compared_paths,
                    "differing_paths": len(declared.authorized_used)
                    + len(declared.unauthorized),
                },
            )
            observed_keys = sorted(
                set(
                    control_manifest["model_visible"]["observation_message"]["payload_keys"]
                )
                | set(
                    control_manifest["model_visible"]["observation_message"][
                        "environment_observation_keys"
                    ]
                )
            )
            checks.record(
                "the_control_model_visible_surface_matches_the_frozen_literal",
                observed_keys == sorted(FROZEN_OBSERVATION_KEYS),
                {"observed": observed_keys, "expected": sorted(FROZEN_OBSERVATION_KEYS)},
            )
            payload_top_level = control_manifest["model_visible"][
                "observation_message"
            ]["payload_keys"]
            checks.record(
                "the_observation_payload_carries_no_duplicated_mirrors",
                sorted(payload_top_level) == sorted(FROZEN_TOP_LEVEL_KEYS),
                {"top_level": sorted(payload_top_level), "expected": FROZEN_TOP_LEVEL_KEYS},
            )
            checks.record(
                "both_arms_report_identical_source_digests",
                control_manifest["host_only"]["source_sha256"]
                == treatment_manifest["host_only"]["source_sha256"],
                {"modules": len(participating_sources())},
            )

            _drive(left, frames)
            _drive(right, frames)
            control_payloads = [
                normalize_request_payload(payload)
                for payload in control_client.request_payloads()
            ]
            treatment_payloads = [
                normalize_request_payload(payload)
                for payload in treatment_client.request_payloads()
            ]
            checks.record(
                "both_arms_issued_the_same_number_of_requests",
                len(control_payloads) == len(treatment_payloads) == len(PROBE_SCRIPT),
                {"control": len(control_payloads), "treatment": len(treatment_payloads)},
            )

            payload_authorized = authorized_request_payload_paths(
                treatment.treatment_capabilities
            )
            observed_reports = []
            for index, (left_payload, right_payload) in enumerate(
                zip(control_payloads, treatment_payloads)
            ):
                report = compare_arms(
                    left_payload, right_payload, authorized_paths=payload_authorized
                )
                observed_reports.append(report.as_json())
                checks.record(
                    f"observed_request_{index}_differs_only_where_authorized",
                    report.ok,
                    report.as_json(),
                )
            checks.record(
                "observed_diff_is_non_empty",
                any(report["authorized_used"] for report in observed_reports),
                {"authorized_used_counts": [
                    len(report["authorized_used"]) for report in observed_reports
                ]},
            )

        # Negative control: an undeclared difference must be rejected.
        smuggled = replace(
            treatment,
            system_prompt=treatment.system_prompt + "\nUndeclared extra instruction.\n",
        )
        smuggled_client = ScriptedChatClient(PROBE_SCRIPT)
        with build_arm(smuggled, run_dir=root / "smuggled", inner_client=smuggled_client) as rogue:
            rogue_report = compare_arms(
                control_manifest,
                arm_manifest(rogue),
                authorized_paths=authorized,
            )
            caught = [item.path for item in rogue_report.unauthorized]
            checks.record(
                "verifier_rejects_an_undeclared_prompt_change",
                "model_visible.system_prompt.sha256" in caught,
                {"unauthorized_paths": caught},
            )
            try:
                assert_authorized(rogue_report, label="negative control")
            except UnauthorizedArmDifference as error:
                checks.record("verifier_raises_on_rejection", True, str(error).splitlines()[0])
            else:
                checks.record("verifier_raises_on_rejection", False, "no exception raised")

    checks.finish(
        artifacts_root() / "p9-0-shared-arm-seam" / "evidence.json",
        control_manifest=control_manifest,
        treatment_manifest=treatment_manifest,
        declared_diff=declared.as_json(),
        observed_request_diffs=observed_reports,
        authorized_manifest_paths=sorted(authorized),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
