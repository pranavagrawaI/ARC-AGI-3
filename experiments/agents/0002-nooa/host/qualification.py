"""Executable qualification inputs shared by 0002-R and 0002-NOOA."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Any, ClassVar


@dataclass(frozen=True)
class PairedRunSpec:
    """The settings that must not drift independently between the two bases."""

    model_profile: str
    provider_endpoint_profile: str
    model_seed: int | None
    environment_profile: str
    environment_seed: int
    max_context_turns: int
    max_recent_transitions: int
    max_model_rounds_per_action: int
    max_decision_repairs: int
    max_model_calls: int
    max_reported_tokens: int
    workspace_timeout_seconds: float
    max_actions: int
    transport_max_retries: int
    transition_observation_policy: str
    include_changed_pixel_counts: bool
    observation_representations: tuple[str, ...]
    coordinate_convention: str
    allowed_python_policy: str
    allowed_distributions: tuple[str, ...]
    host_workspace_files: tuple[str, ...]
    omission_notice_schema: str
    public_status_fields: tuple[str, ...]
    request_profile: str
    legal_event_schema: str
    final_action_schema: str
    host_workspace_initialization: tuple[str, ...]

    _SUPPORTED_SURFACE: ClassVar[dict[str, Any]] = {
        "model_profile": "offline-fixture",
        "provider_endpoint_profile": "offline-fixture",
        "model_seed": None,
        "environment_profile": "offline-public-observation-fixture",
        "environment_seed": 0,
        "max_recent_transitions": 8,
        "transition_observation_policy": (
            "before-and-after-file-pointers-with-public-metadata"
        ),
        "include_changed_pixel_counts": True,
        "observation_representations": (
            "nested-integer-grid",
            "rendered-png",
        ),
        "coordinate_convention": "zero-based-top-left-x-column-y-row",
        "allowed_python_policy": "python-3.12-stdlib-minus-restricted-v1",
        "allowed_distributions": ("numpy", "Pillow"),
        "host_workspace_files": (
            "WORKSPACE.md",
            "NOTES.md",
            "frames.py",
            "public_history.jsonl",
            "observations/",
            "current_observation.json",
            "current_frame.png",
        ),
        "omission_notice_schema": "pointer-only-unsummarized-v1",
        "public_status_fields": (
            "available_actions",
            "full_reset",
            "levels_completed",
            "state",
        ),
        "request_profile": "offline-tool-calling-fixture-v1",
        "legal_event_schema": "integer-id-with-optional-xy-v1",
        "final_action_schema": "action-int-data-object-or-null-reason-string-v1",
        "host_workspace_initialization": (
            "WORKSPACE.md=p9-neutral-workspace-contract-v1",
            "NOTES.md=empty",
            "frames.py=p9-frame-tools-components-false-v1",
            "public_history.jsonl=empty",
            "observations/=empty-directory",
            "current_observation.json=exact-current-public-observation",
            "current_frame.png=exact-current-rendered-public-observation",
        ),
    }

    def __post_init__(self) -> None:
        drift = {
            field: {"supported": supported, "received": getattr(self, field)}
            for field, supported in self._SUPPORTED_SURFACE.items()
            if getattr(self, field) != supported
        }
        if drift:
            raise ValueError(
                "unsupported paired-run surface drift: "
                + json.dumps(drift, sort_keys=True)
            )
        positive = (
            "max_context_turns",
            "max_model_rounds_per_action",
            "max_model_calls",
            "max_reported_tokens",
            "workspace_timeout_seconds",
            "max_actions",
        )
        if any(getattr(self, field) <= 0 for field in positive):
            raise ValueError("paired-run positive budgets must be greater than zero")
        if self.max_decision_repairs < 0 or self.transport_max_retries < 0:
            raise ValueError("paired-run retry budgets must not be negative")

    @classmethod
    def initial(cls) -> "PairedRunSpec":
        return cls(
            model_profile=cls._SUPPORTED_SURFACE["model_profile"],
            provider_endpoint_profile=cls._SUPPORTED_SURFACE[
                "provider_endpoint_profile"
            ],
            model_seed=cls._SUPPORTED_SURFACE["model_seed"],
            environment_profile=cls._SUPPORTED_SURFACE["environment_profile"],
            environment_seed=cls._SUPPORTED_SURFACE["environment_seed"],
            max_context_turns=6,
            max_recent_transitions=cls._SUPPORTED_SURFACE[
                "max_recent_transitions"
            ],
            max_model_rounds_per_action=10,
            max_decision_repairs=2,
            max_model_calls=100,
            max_reported_tokens=250_000,
            workspace_timeout_seconds=20.0,
            max_actions=500,
            transport_max_retries=2,
            transition_observation_policy=cls._SUPPORTED_SURFACE[
                "transition_observation_policy"
            ],
            include_changed_pixel_counts=cls._SUPPORTED_SURFACE[
                "include_changed_pixel_counts"
            ],
            observation_representations=cls._SUPPORTED_SURFACE[
                "observation_representations"
            ],
            coordinate_convention=cls._SUPPORTED_SURFACE[
                "coordinate_convention"
            ],
            allowed_python_policy=cls._SUPPORTED_SURFACE[
                "allowed_python_policy"
            ],
            allowed_distributions=cls._SUPPORTED_SURFACE[
                "allowed_distributions"
            ],
            host_workspace_files=cls._SUPPORTED_SURFACE[
                "host_workspace_files"
            ],
            omission_notice_schema=cls._SUPPORTED_SURFACE[
                "omission_notice_schema"
            ],
            public_status_fields=cls._SUPPORTED_SURFACE[
                "public_status_fields"
            ],
            request_profile=cls._SUPPORTED_SURFACE["request_profile"],
            legal_event_schema=cls._SUPPORTED_SURFACE["legal_event_schema"],
            final_action_schema=cls._SUPPORTED_SURFACE["final_action_schema"],
            host_workspace_initialization=cls._SUPPORTED_SURFACE[
                "host_workspace_initialization"
            ],
        )

    def control_overrides(self) -> dict[str, int | float]:
        """Project only fields accepted by the frozen 0002-R arm factory."""

        return {
            "max_actions": self.max_actions,
            "max_context_turns": self.max_context_turns,
            "max_decision_repairs": self.max_decision_repairs,
            "max_model_calls": self.max_model_calls,
            "max_model_rounds_per_action": self.max_model_rounds_per_action,
            "max_reported_tokens": self.max_reported_tokens,
            "transport_max_retries": self.transport_max_retries,
            "workspace_timeout_seconds": self.workspace_timeout_seconds,
        }

    def as_record(self) -> dict[str, Any]:
        """Return the canonical IPC/audit projection of the shared settings."""

        return json.loads(json.dumps(asdict(self), sort_keys=True))


def verify_source_manifest() -> dict[str, Any]:
    """Verify pinned source bytes and every destination claimed as unmodified."""

    control_root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (control_root / "SOURCE_MANIFEST.json").read_text(encoding="utf-8")
    )
    upstream = manifest["upstream"]
    checkout = Path(upstream["local_reference_checkout"])
    commit = upstream["commit"]
    mismatches: list[str] = []
    verified_sources = 0

    for source in manifest["sources"]:
        source_path = source["source"]
        expected = source["sha256"]
        completed = subprocess.run(
            ["git", "-C", str(checkout), "show", f"{commit}:{source_path}"],
            check=False,
            capture_output=True,
        )
        if completed.returncode:
            mismatches.append(
                f"cannot read upstream {commit}:{source_path}: "
                f"{completed.stderr.decode(errors='replace').strip()}"
            )
            continue
        actual = sha256(completed.stdout).hexdigest()
        if actual != expected:
            mismatches.append(
                f"upstream digest mismatch for {source_path}: "
                f"expected {expected}, got {actual}"
            )
            continue
        verified_sources += 1

        status = source["copy_status"]
        if "unmodified" in status or "verbatim" in status:
            destination = control_root / source["planned_destination"]
            if not destination.is_file():
                mismatches.append(f"missing copied destination: {destination}")
                continue
            copied = sha256(destination.read_bytes()).hexdigest()
            if copied != expected:
                mismatches.append(
                    f"copied digest mismatch for {destination}: "
                    f"expected {expected}, got {copied}"
                )

    for destination in ("nooa-runtime/agent.py", "nooa-runtime/launcher.py"):
        first_lines = (
            (control_root / destination).read_text(encoding="utf-8").splitlines()[:12]
        )
        if not any("SPDX-FileCopyrightText" in line for line in first_lines):
            mismatches.append(f"derived source lacks copyright notice: {destination}")
        if not any("Derived by copy-and-prune" in line for line in first_lines):
            mismatches.append(f"derived source lacks modification notice: {destination}")

    return {
        "schema": "0002-nooa-source-verification-v1",
        "upstream_commit": commit,
        "verified_sources": verified_sources,
        "mismatches": mismatches,
    }


def qualify_model_visible_parity(
    *,
    run_root: Path,
    spec: PairedRunSpec,
) -> dict[str, Any]:
    """Normalize one observed NOOA crossing against the frozen 0002-R surface."""

    control = _control_model_visible(spec)
    process_records = [
        json.loads(line)
        for line in (run_root / "process.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    crossing = next(record for record in process_records if record["kind"] == "model_crossing")
    call = crossing["audit"]["model_calls"][0]
    messages = call["messages"]
    system = messages[0]
    observation_message = next(
        message for message in messages if isinstance(message.get("content"), list)
    )
    content = observation_message["content"]
    payload = json.loads(content[0]["text"])
    tools = call["tools"]
    unmatched: list[str] = []

    expected_payload_keys = set(control["observation_message"]["payload_keys"])
    if set(payload) != expected_payload_keys:
        unmatched.append(
            "observation payload keys differ: "
            f"control={sorted(expected_payload_keys)!r} nooa={sorted(payload)!r}"
        )
    expected_environment_keys = set(
        control["observation_message"]["environment_observation_keys"]
    )
    actual_environment = payload.get("environment_observation", {})
    if set(actual_environment) != expected_environment_keys:
        unmatched.append(
            "environment observation keys differ: "
            f"control={sorted(expected_environment_keys)!r} "
            f"nooa={sorted(actual_environment)!r}"
        )
    if [part.get("type") for part in content] != ["text", "image_url"]:
        unmatched.append("NOOA does not expose exactly one text payload and one current image")

    control_python = control["tools"][0]["function"]
    nooa_python = next((tool for tool in tools if tool.get("name") == "execute_python"), None)
    if nooa_python is None:
        unmatched.append("NOOA has no generated-Python tool")
    else:
        if nooa_python.get("description") != control_python.get("description"):
            unmatched.append("generated-Python tool description differs semantically")
        if nooa_python.get("parameters") != control_python.get("parameters"):
            unmatched.append(
                "generated-Python argument schema differs: "
                f"control={control_python.get('parameters')!r} "
                f"nooa={nooa_python.get('parameters')!r}"
            )

    return_tool = next((tool for tool in tools if tool.get("name") == "return_result"), None)
    result_properties = (
        return_tool.get("parameters", {})
        .get("properties", {})
        .get("result", {})
        .get("properties", {})
        if return_tool
        else {}
    )
    if list(result_properties) != control["decision_contract"]["fields"]:
        unmatched.append("final-action fields differ")

    required_workspace = set(control["workspace"]["reserved_paths"])
    actual_workspace = {path.name for path in (run_root / "workspace").iterdir()}
    missing_workspace = sorted(required_workspace - actual_workspace)
    if missing_workspace:
        unmatched.append(f"required shared workspace entries missing: {missing_workspace!r}")

    lowered_system = str(system.get("content", "")).lower()
    leaked_terms = [
        term
        for term in (
            "nooa",
            "agent instance",
            "self.",
            "runtime",
            "provider",
            "controller",
            "context",
            "trace",
            "ipc",
            "process",
            "host",
            "context_manager",
            "event_manager",
        )
        if term in lowered_system
    ]
    if leaked_terms:
        unmatched.append(f"system prompt exposes runtime terms: {leaked_terms!r}")

    report = {
        "schema": "0002-nooa-model-visible-parity-v1",
        "control_arm": "0002-R",
        "candidate_arm": "0002-NOOA",
        "same_capability_different_transport": [
            {
                "path": "tools.python.name",
                "control": "python",
                "nooa": "execute_python",
            },
            {
                "path": "decision.transport",
                "control": "assistant JSON parsed by host",
                "nooa": "typed return_result tool validated by host",
            },
        ],
        "measured_substrate_differences": [
            {
                "path": "runtime",
                "control": "Python 3.14 in shared arm process",
                "nooa": "Python 3.12 worker over 0002-nooa-ipc-v1",
            }
        ],
        "unmatched_affordances": unmatched,
    }
    (run_root / "parity-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _control_model_visible(spec: PairedRunSpec) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[4]
    baseline = repo / "experiments" / "agents" / "0002-llm-repl-agent"
    p9_parent = repo / "experiments" / "agents" / "0006-capability-environment"
    for path in (baseline, p9_parent):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from p9.arms import arm_manifest, build_arm, control_spec

    class SilentClient:
        def complete(self, *_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("manifest construction must not call the model")

    with TemporaryDirectory(prefix="0002-nooa-control-manifest-") as directory:
        with build_arm(
            control_spec(**spec.control_overrides()),
            run_dir=Path(directory) / "run",
            inner_client=SilentClient(),
        ) as arm:
            return arm_manifest(arm)["model_visible"]
