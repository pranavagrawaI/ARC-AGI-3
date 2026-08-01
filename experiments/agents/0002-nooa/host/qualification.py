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
from typing import Any, ClassVar, Mapping


@dataclass(frozen=True, slots=True)
class ProviderRequestProfile:
    """Credential-free commitment to every static provider request field."""

    schema: str
    api_surface: str
    model: str
    wire_model: str
    api_base: str
    max_output_tokens: int
    reasoning_effort: str | None
    request_timeout_seconds: float
    temperature: float | None
    seed: int | None
    stream: bool
    drop_params: bool
    parallel_tool_calls: bool
    tool_choice: str
    truncation: str
    nooa_max_retries: int
    nooa_rate_limit_extra_retries: int
    litellm_num_retries: int

    @classmethod
    def responses(
        cls,
        *,
        model: str,
        wire_model: str | None = None,
        api_base: str,
        max_output_tokens: int,
        reasoning_effort: str | None,
        request_timeout_seconds: float,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> "ProviderRequestProfile":
        return cls(
            schema="0002-nooa-provider-request-v2",
            api_surface="responses",
            model=model,
            wire_model=wire_model or model,
            api_base=api_base,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            request_timeout_seconds=request_timeout_seconds,
            temperature=temperature,
            seed=seed,
            stream=False,
            drop_params=True,
            parallel_tool_calls=False,
            tool_choice="auto",
            truncation="disabled",
            nooa_max_retries=0,
            nooa_rate_limit_extra_retries=0,
            litellm_num_retries=0,
        )

    def __post_init__(self) -> None:
        fixed = {
            "schema": "0002-nooa-provider-request-v2",
            "api_surface": "responses",
            "stream": False,
            "drop_params": True,
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "truncation": "disabled",
            "nooa_max_retries": 0,
            "nooa_rate_limit_extra_retries": 0,
            "litellm_num_retries": 0,
        }
        drift = {
            field: {"required": required, "received": getattr(self, field)}
            for field, required in fixed.items()
            if type(getattr(self, field)) is not type(required)
            or getattr(self, field) != required
        }
        if drift:
            raise ValueError(
                "unsupported provider request profile: "
                + json.dumps(drift, sort_keys=True)
            )
        if not all(
            isinstance(value, str)
            for value in (self.model, self.wire_model, self.api_base)
        ):
            raise ValueError("provider models and api_base must be strings")
        if not self.model or not self.wire_model or not self.api_base:
            raise ValueError("provider models and api_base must be non-empty")
        if type(self.max_output_tokens) is not int or self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be a positive integer")
        if (
            isinstance(self.request_timeout_seconds, bool)
            or not isinstance(self.request_timeout_seconds, (int, float))
            or self.request_timeout_seconds <= 0
        ):
            raise ValueError("request_timeout_seconds must be positive")
        if self.seed is not None and type(self.seed) is not int:
            raise ValueError("seed must be an integer or null")
        if self.reasoning_effort is not None and not isinstance(
            self.reasoning_effort,
            str,
        ):
            raise ValueError("reasoning_effort must be a string or null")
        if self.temperature is not None and (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
        ):
            raise ValueError("temperature must be numeric or null")


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
    usability_probe_instruction: str | None = None
    provider_request_profile: ProviderRequestProfile | None = None

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
            "WORKSPACE.md=direct-import-contract-v1",
            "NOTES.md=empty",
            "frames.py=direct-import-frame-tools-v1",
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
            if field
            not in {
                "host_workspace_initialization",
                "model_profile",
                "provider_endpoint_profile",
                "model_seed",
                "environment_profile",
                "environment_seed",
                "request_profile",
            }
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
        if self.usability_probe_instruction is not None and not isinstance(
            self.usability_probe_instruction,
            str,
        ):
            raise ValueError("usability_probe_instruction must be a string or null")
        expected_workspace_initialization = self.workspace_initialization_for(
            self.usability_probe_instruction
        )
        if self.host_workspace_initialization != expected_workspace_initialization:
            raise ValueError(
                "workspace initialization does not commit the usability probe: "
                + json.dumps(
                    {
                        "expected": expected_workspace_initialization,
                        "received": self.host_workspace_initialization,
                    },
                    sort_keys=True,
                )
            )
        profile = self.provider_request_profile
        if profile is None:
            expected_projection = {
                field: self._SUPPORTED_SURFACE[field]
                for field in (
                    "model_profile",
                    "provider_endpoint_profile",
                    "model_seed",
                    "environment_profile",
                    "environment_seed",
                    "request_profile",
                )
            }
            message = "unsupported paired-run surface drift"
        else:
            expected_projection = {
                "model_profile": profile.model,
                "provider_endpoint_profile": profile.api_base,
                "model_seed": profile.seed,
                "request_profile": profile.schema,
            }
            message = "provider profile projection drift"
            if not self.environment_profile:
                raise ValueError("provider run environment_profile must be non-empty")
            if type(self.environment_seed) is not int:
                raise ValueError("provider run environment_seed must be an integer")
        projection_drift = {
            field: {"expected": expected, "received": getattr(self, field)}
            for field, expected in expected_projection.items()
            if getattr(self, field) != expected
        }
        if projection_drift:
            raise ValueError(
                message + ": " + json.dumps(projection_drift, sort_keys=True)
            )

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

    @classmethod
    def workspace_initialization_for(
        cls,
        instruction: str | None,
    ) -> tuple[str, ...]:
        initialization = list(cls._SUPPORTED_SURFACE["host_workspace_initialization"])
        if instruction is not None:
            notes_commitment = "NOTES.md=sha256:" + sha256(
                instruction.encode("utf-8")
            ).hexdigest()
            initialization[initialization.index("NOTES.md=empty")] = notes_commitment
        return tuple(initialization)

    @classmethod
    def provider(
        cls,
        profile: ProviderRequestProfile,
        *,
        environment_profile: str,
        environment_seed: int,
    ) -> "PairedRunSpec":
        """Build the live-provider variant while retaining all shared budgets."""

        offline = cls.initial()
        values = asdict(offline)
        values.update(
            model_profile=profile.model,
            provider_endpoint_profile=profile.api_base,
            model_seed=profile.seed,
            environment_profile=environment_profile,
            environment_seed=environment_seed,
            request_profile=profile.schema,
            provider_request_profile=profile,
        )
        return cls(**values)

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

    for destination in (
        "nooa-runtime/agent.py",
        "nooa-runtime/launcher.py",
        "nooa-runtime/llm.py",
    ):
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


_FORBIDDEN_MODEL_AFFORDANCES = {
    "runtime_introspection": (
        "nooa",
        "agent instance",
        "context_manager",
        "event_manager",
        "provider client",
        "ipc transport",
        "trace store",
        "process supervisor",
    ),
    "hidden_history_query": ("trajectory()", "hidden event history", "event history query"),
    "automatic_retrieval": (
        "memoryskill",
        "automatic retrieval",
        "spontaneous recall",
        "hybrid retrieval",
        "memory graph",
    ),
    "summarization_or_reflection": (
        "generic summarization",
        "automatic summarization",
        "reflection pass",
        "consolidation pass",
    ),
    "semantic_skill_text": ("arc solver skill", "world-model coaching", "semantic skill"),
    "history_selection": ("history selection", "select relevant history"),
    "action_batching": ("submit_actions", "action batch", "multiple environment actions"),
    "host_force_advance": ("force-advance", "force advance"),
}

_EXPECTED_NOOA_SYSTEM = (
    "You choose one public environment action at a time. The exact public\n"
    "action point is available as `action_point`. Use `execute_python(code)` for generic Python\n"
    "and relative workspace files. Variables and files persist across action points for this run.\n"
    "Finish with `return_result({\"action\": integer, \"data\": null-or-object, "
    "\"reason\": string})`.\n"
    "Only currently legal action IDs are accepted. Integer event IDs carry no semantic meaning.\n"
    "Plain text does not finish an action point."
)


def detect_unmatched_model_affordances(value: Any) -> list[str]:
    """Name forbidden cognitive/runtime affordances in a model-visible value."""

    lowered = json.dumps(value, sort_keys=True).lower()
    return [
        name
        for name, markers in _FORBIDDEN_MODEL_AFFORDANCES.items()
        if any(marker in lowered for marker in markers)
    ]


def _host_supplied_model_surface(
    model_calls: list[dict[str, Any]],
    *,
    workspace_contract: str,
) -> dict[str, Any]:
    """Retain only provenance-known host instructions, observations, and tools."""

    messages: list[dict[str, Any]] = []
    tools: list[Any] = []
    for call in model_calls:
        tools.append(call.get("tools", []))
        for message in call.get("messages", []):
            role = message.get("role")
            content = message.get("content")
            if role == "system":
                messages.append({"role": role, "content": content})
            elif role == "user" and isinstance(content, list):
                sanitized = json.loads(json.dumps(content))
                try:
                    payload = json.loads(sanitized[0]["text"])
                except (IndexError, KeyError, TypeError, json.JSONDecodeError):
                    pass
                else:
                    notes = payload.get("notes")
                    if isinstance(notes, dict) and "content" in notes:
                        notes["content"] = "<model-authored-notes>"
                    sanitized[0]["text"] = json.dumps(
                        payload,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                messages.append({"role": role, "content": sanitized})
            elif (
                role == "user"
                and isinstance(content, str)
                and content.startswith("Transcript omission.")
            ):
                messages.append({"role": role, "content": content})
    return {
        "messages": messages,
        "tools": tools,
        "workspace_contract": workspace_contract,
    }


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _redact_data_images(value: Any) -> Any:
    if isinstance(value, list):
        return [_redact_data_images(item) for item in value]
    if not isinstance(value, dict):
        return value
    image = value.get("image_url")
    if isinstance(image, dict):
        url = image.get("url")
        if isinstance(url, str) and url.startswith("data:image/"):
            return {
                **value,
                "image_url": {
                    "redacted": True,
                    "sha256": sha256(url.encode("utf-8")).hexdigest(),
                    "chars": len(url),
                },
            }
    return {key: _redact_data_images(item) for key, item in value.items()}


def _observation_payload(messages: list[dict[str, Any]]) -> dict[str, Any]:
    message = next(
        item for item in reversed(messages) if isinstance(item.get("content"), list)
    )
    return json.loads(message["content"][0]["text"])


def _omission_notices(messages: list[dict[str, Any]]) -> list[str]:
    return [
        content
        for message in messages
        if isinstance((content := message.get("content")), str)
        and content.startswith("Transcript omission.")
    ]


def _host_message_shape(messages: list[dict[str, Any]]) -> list[str]:
    shape = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            shape.append("system")
        elif role == "user" and isinstance(content, list):
            shape.append("public_observation")
        elif (
            role == "user"
            and isinstance(content, str)
            and content.startswith("Transcript omission.")
        ):
            shape.append("pointer_only_omission")
        elif role == "user":
            shape.append("other_host_user_message")
    return shape


def _candidate_tool_contract_errors(tools: Any) -> list[str]:
    if not isinstance(tools, list):
        return ["tool catalogue is not a list"]
    names = [tool.get("name") for tool in tools if isinstance(tool, Mapping)]
    errors = []
    if names != ["execute_python", "return_result"]:
        errors.append(f"tool names changed: {names!r}")
    execute = next(
        (
            tool
            for tool in tools
            if isinstance(tool, Mapping) and tool.get("name") == "execute_python"
        ),
        None,
    )
    execute_properties = (
        execute.get("parameters", {}).get("properties", {})
        if isinstance(execute, Mapping)
        else {}
    )
    if set(execute_properties) != {"code"}:
        errors.append("execute_python arguments changed")
    result = next(
        (
            tool
            for tool in tools
            if isinstance(tool, Mapping) and tool.get("name") == "return_result"
        ),
        None,
    )
    result_properties = (
        result.get("parameters", {})
        .get("properties", {})
        .get("result", {})
        .get("properties", {})
        if isinstance(result, Mapping)
        else {}
    )
    if set(result_properties) != {"action", "data", "reason"}:
        errors.append("return_result decision fields changed")
    return errors


def _normalize_omission_notice(notice: str) -> str:
    prefix = "Transcript omission. Conversation turns for action steps "
    suffix = " are no longer in this context"
    if not notice.startswith(prefix) or suffix not in notice:
        return notice
    _steps, remainder = notice[len(prefix) :].split(suffix, 1)
    return prefix + "{steps}" + suffix + remainder


def _control_fixture_model_visible(
    *,
    spec: PairedRunSpec,
    run_root: Path,
    process_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run 0002-R over the exact successful candidate observation sequence."""

    repo = Path(__file__).resolve().parents[4]
    baseline = repo / "experiments" / "agents" / "0002-llm-repl-agent"
    p9_parent = repo / "experiments" / "agents" / "0006-capability-environment"
    for path in (baseline, p9_parent):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import numpy as np

    from experiments.harness import Observation, Transition
    from host.workspace import (
        adapt_control_spec,
        adapt_workspace_for_direct_import,
        direct_import_control_runtime,
    )
    from p9.arms import arm_manifest, build_arm, control_spec
    from p9.fixtures import ScriptedChatClient, action_turn

    crossings = sorted(
        (record for record in process_records if record.get("kind") == "model_crossing"),
        key=lambda record: record["action_point_id"],
    )
    observations: list[Observation] = []
    script: list[dict[str, Any]] = []
    requests_by_step = {
        record["action_point_id"]: record
        for record in process_records
        if record.get("kind") == "action_request"
    }
    for crossing in crossings:
        step = crossing["action_point_id"]
        request_record = requests_by_step.get(step)
        if not isinstance(request_record, Mapping):
            raise ValueError(f"candidate crossing {step} has no host action request")
        host_observation = request_record.get("host_observation")
        if not isinstance(host_observation, Mapping):
            raise ValueError(f"candidate crossing {step} has no host observation evidence")
        relative_path = host_observation.get("path")
        commitment = host_observation.get("sha256")
        if not isinstance(relative_path, str) or not isinstance(commitment, str):
            raise ValueError(f"candidate crossing {step} has malformed host observation evidence")
        observation_bytes = (run_root / relative_path).read_bytes()
        actual_commitment = sha256(observation_bytes).hexdigest()
        if actual_commitment != commitment:
            raise ValueError(
                f"candidate crossing {step} host observation commitment mismatch"
            )
        record = json.loads(observation_bytes)
        pixels = np.asarray(record["pixels"], dtype=int)
        pixels.setflags(write=False)
        observations.append(
            Observation(
                pixels=pixels,
                available_actions=tuple(record["available_actions"]),
                levels_completed=record["levels_completed"],
                state=record["state"],
                full_reset=record["full_reset"],
            )
        )
        decision = crossing.get("worker_decision")
        if not isinstance(decision, Mapping):
            raise ValueError(f"candidate crossing {step} has no successful decision")
        script.append(
            action_turn(
                int(decision["action"]),
                data=decision.get("data"),
                reason=str(decision.get("reason", "")),
            )
        )

    client = ScriptedChatClient(script, model=spec.model_profile)
    with TemporaryDirectory(prefix="0002-nooa-control-fixture-") as directory:
        with direct_import_control_runtime():
            with build_arm(
                adapt_control_spec(control_spec(**spec.control_overrides())),
                run_dir=Path(directory) / "run",
                inner_client=client,
            ) as arm:
                adapt_workspace_for_direct_import(arm.workspace.root)
                manifest = arm_manifest(arm)["model_visible"]
                manifest["workspace"]["provided_helper_sha256"] = sha256(
                    (arm.workspace.root / "frames.py").read_bytes()
                ).hexdigest()
                decisions = []
                for index, before in enumerate(observations):
                    decision = arm.agent.act(before)
                    decisions.append(decision)
                    if index + 1 < len(observations):
                        arm.agent.observe(
                            Transition(
                                before=before,
                                decision=decision,
                                after=observations[index + 1],
                            )
                        )
                workspace_contract = (
                    arm.workspace.root / "WORKSPACE.md"
                ).read_text(encoding="utf-8")
                requests = client.request_payloads()
    return {
        "manifest": manifest,
        "requests": requests,
        "workspace_contract": workspace_contract,
        "actions": len(decisions),
    }


def qualify_model_visible_parity(
    *,
    run_root: Path,
    spec: PairedRunSpec,
) -> dict[str, Any]:
    """Run 0002-R over the candidate fixture and diff every model-facing crossing."""

    process_records = [
        json.loads(line)
        for line in (run_root / "process.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    crossings = sorted(
        (record for record in process_records if record.get("kind") == "model_crossing"),
        key=lambda record: record["action_point_id"],
    )
    if not crossings:
        raise ValueError("candidate run contains no model crossing")
    control_fixture = _control_fixture_model_visible(
        spec=spec,
        run_root=run_root,
        process_records=process_records,
    )
    control = control_fixture["manifest"]
    control_request = control_fixture["requests"][-1]
    control_messages = control_request["messages"]
    control_payload = _observation_payload(control_messages)
    crossing = crossings[-1]
    call = crossing["audit"]["model_calls"][-1]
    messages = call["messages"]
    payload = _observation_payload(messages)
    tools = call["tools"]
    system = messages[0]
    candidate_workspace_contract = (run_root / "workspace" / "WORKSPACE.md").read_text(
        encoding="utf-8"
    )

    catalogue: list[dict[str, Any]] = []

    def compare(
        path: str,
        control_value: Any,
        candidate_value: Any,
        *,
        difference_class: str = "unmatched_affordance",
        evidence: str,
    ) -> None:
        classification = (
            "same_capability_different_transport"
            if control_value == candidate_value
            else difference_class
        )
        catalogue.append(
            {
                "path": path,
                "classification": classification,
                "control": control_value,
                "nooa": candidate_value,
                "evidence": evidence,
            }
        )

    control_system = str(control_messages[0].get("content", ""))
    candidate_system = str(system.get("content", ""))
    required_system_fragments = (
        "one public environment action",
        "exact public action point",
        "generic python",
        "files persist",
        "return_result",
        "legal action ids",
        "no semantic meaning",
        "plain text does not finish",
    )
    normalized_candidate_system = " ".join(candidate_system.lower().split())
    missing_system_semantics = [
        fragment
        for fragment in required_system_fragments
        if fragment not in normalized_candidate_system
    ]
    compare(
        "system_prompt.semantic_sections",
        [],
        missing_system_semantics,
        evidence="required neutral action, Python/files, legality, and event-ID clauses",
    )
    catalogue.append(
        {
            "path": "system_prompt.digest",
            "classification": "measured_substrate_difference",
            "control": {
                "sha256": sha256(control_system.encode("utf-8")).hexdigest(),
                "chars": len(control_system),
            },
            "nooa": {
                "sha256": sha256(candidate_system.encode("utf-8")).hexdigest(),
                "chars": len(candidate_system),
            },
            "evidence": "different native instruction transports with separately committed bytes",
        }
    )
    compare(
        "system_prompt.nooa_exact",
        _EXPECTED_NOOA_SYSTEM,
        candidate_system,
        evidence="frozen candidate system instruction bytes",
    )

    compare(
        "observation.payload_fields",
        sorted(control_payload),
        sorted(payload),
        evidence="latest actual request payload",
    )
    control_environment = control_payload.get("environment_observation", {})
    candidate_environment = payload.get("environment_observation", {})
    compare(
        "observation.environment_fields",
        sorted(control_environment),
        sorted(candidate_environment),
        evidence="latest actual request payload",
    )
    compare(
        "observation.representations",
        control_environment.get("representations"),
        candidate_environment.get("representations"),
        evidence="text-grid and rendered-image descriptors from the same observation",
    )
    compare(
        "observation.public_status_fields",
        sorted(spec.public_status_fields),
        sorted(
            field
            for field in candidate_environment.get("current_metadata", {})
            if field in spec.public_status_fields
        ),
        evidence="shared PairedRunSpec public-status catalogue",
    )
    control_content = next(
        message["content"]
        for message in reversed(control_messages)
        if isinstance(message.get("content"), list)
    )
    candidate_content = next(
        message["content"]
        for message in reversed(messages)
        if isinstance(message.get("content"), list)
    )
    compare(
        "observation.content_parts",
        [part.get("type") for part in control_content],
        [part.get("type") for part in candidate_content],
        evidence="one current text observation and one current rendered image",
    )
    control_image = _redact_data_images(control_content[1])["image_url"]
    candidate_image = candidate_content[1].get("image_url")
    compare(
        "observation.current_image_commitment",
        control_image,
        candidate_image,
        evidence="same annotated PNG, body redacted after hashing",
    )

    control_python = control["tools"][0]["function"]
    nooa_python = next((tool for tool in tools if tool.get("name") == "execute_python"), None)
    compare(
        "tools.python.description",
        control_python.get("description"),
        nooa_python.get("description") if nooa_python else None,
        evidence="constructed control manifest versus candidate actual tool crossing",
    )
    compare(
        "tools.python.arguments",
        control_python.get("parameters"),
        nooa_python.get("parameters") if nooa_python else None,
        evidence="constructed control manifest versus candidate actual tool crossing",
    )
    catalogue.append(
        {
            "path": "python.builtins",
            "classification": "measured_substrate_difference",
            "control": "0002-R isolated Python builtins",
            "nooa": {
                "generic_object_introspection": ["dir", "getattr"],
                "nooa_validator_restrictions": [
                    "compile",
                    "eval",
                    "exec",
                    "globals",
                    "locals",
                    "vars",
                    "__import__",
                ],
            },
            "evidence": "exercised namespace-pruning and generic-object-introspection fixtures",
        }
    )
    compare(
        "python.packages",
        sorted(name.lower() for name in control["workspace"]["allowed_distributions"]),
        sorted(name.lower() for name in spec.allowed_distributions),
        evidence="exercised NumPy/Pillow candidate fixture and constructed control manifest",
    )

    required_workspace = set(control["workspace"]["reserved_paths"])
    actual_workspace = {path.name for path in (run_root / "workspace").iterdir()}
    compare(
        "workspace.required_files",
        [],
        sorted(required_workspace - actual_workspace),
        evidence="actual candidate workspace after fixture run",
    )
    compare(
        "workspace.contract_digest",
        sha256(control_fixture["workspace_contract"].encode("utf-8")).hexdigest(),
        sha256(candidate_workspace_contract.encode("utf-8")).hexdigest(),
        evidence="exact model-readable WORKSPACE.md bytes",
    )
    compare(
        "workspace.helper_digest",
        control["workspace"]["provided_helper_sha256"],
        sha256((run_root / "workspace" / "frames.py").read_bytes()).hexdigest(),
        evidence="exact model-readable frames.py bytes",
    )
    compare(
        "workspace.timeout_seconds",
        control["workspace"]["timeout_seconds"],
        spec.workspace_timeout_seconds,
        evidence="constructed control and immutable candidate PairedRunSpec",
    )

    def visible_steps(candidate_messages: list[dict[str, Any]]) -> list[int]:
        steps: list[int] = []
        for message in candidate_messages:
            if not isinstance(message.get("content"), list):
                continue
            try:
                step = json.loads(message["content"][0]["text"]).get("step")
            except (KeyError, TypeError, json.JSONDecodeError):
                continue
            if type(step) is int:
                steps.append(step)
        return steps

    compare(
        "context.recent_turn_boundaries",
        visible_steps(control_messages),
        visible_steps(messages),
        evidence="ordered observation-bearing turns in the latest actual request",
    )
    control_omissions = _omission_notices(control_messages)
    candidate_omissions = _omission_notices(messages)
    compare(
        "context.omission_notice",
        [_normalize_omission_notice(notice) for notice in control_omissions],
        [_normalize_omission_notice(notice) for notice in candidate_omissions],
        evidence="actual latest request when eviction has occurred; empty before that boundary",
    )
    if control_omissions or candidate_omissions:
        catalogue.append(
            {
                "path": "context.omission_step_index",
                "classification": "measured_substrate_difference",
                "control": control_omissions,
                "nooa": candidate_omissions,
                "evidence": (
                    "0002-R labels completed action turns from 1; NOOA labels "
                    "public action points from 0"
                ),
            }
        )
    compare(
        "context.transition_boundary",
        min(len(crossings) - 1, spec.max_recent_transitions),
        len(candidate_environment.get("recent_transitions", [])),
        evidence="latest bounded public transition projection",
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
    compare(
        "decision.fields",
        control["decision_contract"]["fields"],
        list(result_properties),
        evidence="assistant JSON versus typed return_result transport",
    )
    catalogue.extend(
        [
            {
                "path": "decision.transport",
                "classification": "same_capability_different_transport",
                "control": "assistant JSON parsed by host",
                "nooa": "typed return_result validated again by host",
                "evidence": "one legal action object crosses either substrate",
            },
            {
                "path": "decision.temporal_seam",
                "classification": "same_capability_different_transport",
                "control": "synchronous act returns one accepted Decision",
                "nooa": "action-point closure plus host workspace commitment",
                "evidence": "replacement and post-accept workspace mutants are rejected",
            },
            {
                "path": "errors_and_recovery",
                "classification": "measured_substrate_difference",
                "control": control["failure_taxonomy"],
                "nooa": {
                    "python": ["syntax", "runtime", "truncation", "timeout_restart"],
                    "decision": ["malformed", "illegal", "coordinates"],
                    "transport": ["retryable", "fatal", "uncertain"],
                    "process": ["death", "ipc_timeout"],
                },
                "evidence": "G6 injected failure matrix",
            },
        ]
    )
    compare(
        "budgets.deliberation",
        control["deliberation_budgets"],
        {
            "max_model_rounds_per_action": spec.max_model_rounds_per_action,
            "max_decision_repairs": spec.max_decision_repairs,
            "action_required_on_final_round": True,
        },
        evidence="constructed control and immutable candidate PairedRunSpec",
    )
    compare(
        "budgets.run",
        {
            "max_actions": spec.max_actions,
            "max_model_calls": spec.max_model_calls,
            "max_reported_tokens": spec.max_reported_tokens,
        },
        {
            "max_actions": spec.max_actions,
            "max_model_calls": spec.max_model_calls,
            "max_reported_tokens": spec.max_reported_tokens,
        },
        evidence="one immutable PairedRunSpec projected to both arms",
    )

    control_request_sha256 = sha256(_canonical_bytes(control_request)).hexdigest()
    candidate_request_sha256 = call.get("canonical_request_sha256")
    catalogue.append(
        {
            "path": "request.canonical_content",
            "classification": "measured_substrate_difference",
            "control": {
                "sha256": control_request_sha256,
                "encoding": "json-utf8-sort-keys-compact-v1",
            },
            "nooa": {
                "sha256": candidate_request_sha256,
                "encoding": call.get("canonicalization"),
            },
            "evidence": "exact requests retained below with image bodies replaced by commitments",
        }
    )
    control_actual = []
    candidate_actual = call.get("actual_provider_crossings", [])
    catalogue.append(
        {
            "path": "request.actual_provider_payload",
            "classification": "measured_substrate_difference",
            "control": {"available": bool(control_actual), "crossings": control_actual},
            "nooa": {
                "available": bool(candidate_actual),
                "crossings": candidate_actual,
            },
            "evidence": (
                "offline fixtures have no HTTP crossing; provider runs retain "
                "redacted HTTP evidence"
            ),
        }
    )

    candidate_calls = [
        model_call
        for candidate_crossing in crossings
        for model_call in candidate_crossing["audit"]["model_calls"]
    ]
    control_calls = control_fixture["requests"]
    compare(
        "crossings.model_call_count",
        len(control_calls),
        len(candidate_calls),
        evidence="one corresponding actual model call per fixture action point",
    )
    candidate_tool_baseline = candidate_calls[0].get("tools") if candidate_calls else None
    for index in range(max(len(control_calls), len(candidate_calls))):
        if index >= len(control_calls) or index >= len(candidate_calls):
            continue
        control_call = control_calls[index]
        candidate_call = candidate_calls[index]
        control_call_messages = control_call["messages"]
        candidate_call_messages = candidate_call["messages"]
        control_call_payload = _observation_payload(control_call_messages)
        candidate_call_payload = _observation_payload(candidate_call_messages)
        prefix = f"crossings.call_{index}"
        candidate_call_system = " ".join(
            str(candidate_call_messages[0].get("content", "")).lower().split()
        )
        compare(
            f"{prefix}.system_semantics",
            [],
            [
                fragment
                for fragment in required_system_fragments
                if fragment not in candidate_call_system
            ],
            evidence="required neutral system clauses at this crossing",
        )
        compare(
            f"{prefix}.system_exact",
            _EXPECTED_NOOA_SYSTEM,
            str(candidate_call_messages[0].get("content", "")),
            evidence="frozen candidate system instruction bytes at this crossing",
        )
        compare(
            f"{prefix}.host_message_shape",
            _host_message_shape(control_call_messages),
            _host_message_shape(candidate_call_messages),
            evidence="host observation and pointer-only omission sequence",
        )
        compare(
            f"{prefix}.payload_fields",
            sorted(control_call_payload),
            sorted(candidate_call_payload),
            evidence="corresponding actual request payload keys",
        )
        control_call_environment = control_call_payload.get("environment_observation", {})
        candidate_call_environment = candidate_call_payload.get(
            "environment_observation",
            {},
        )
        compare(
            f"{prefix}.environment_fields",
            sorted(control_call_environment),
            sorted(candidate_call_environment),
            evidence="corresponding actual environment-observation keys",
        )
        for field in sorted(
            set(control_call_environment) | set(candidate_call_environment)
        ):
            compare(
                f"{prefix}.environment.{field}",
                control_call_environment.get(field),
                candidate_call_environment.get(field),
                evidence="corresponding public observation value",
            )
        for field in ("step", "deliberation"):
            compare(
                f"{prefix}.payload.{field}",
                control_call_payload.get(field),
                candidate_call_payload.get(field),
                evidence="corresponding host-authored request value",
            )
        control_notes = control_call_payload.get("notes", {})
        candidate_notes = candidate_call_payload.get("notes", {})
        compare(
            f"{prefix}.payload.notes_contract",
            {
                "keys": sorted(control_notes),
                "file": control_notes.get("file"),
                "limit": control_notes.get("limit"),
            },
            {
                "keys": sorted(candidate_notes),
                "file": candidate_notes.get("file"),
                "limit": candidate_notes.get("limit"),
            },
            evidence="model-authored note content is excluded; host envelope is compared",
        )
        control_call_content = next(
            message["content"]
            for message in reversed(control_call_messages)
            if isinstance(message.get("content"), list)
        )
        candidate_call_content = next(
            message["content"]
            for message in reversed(candidate_call_messages)
            if isinstance(message.get("content"), list)
        )
        compare(
            f"{prefix}.content_parts",
            [part.get("type") for part in control_call_content],
            [part.get("type") for part in candidate_call_content],
            evidence="corresponding text/image observation envelope",
        )
        compare(
            f"{prefix}.current_image_commitment",
            _redact_data_images(control_call_content[1]).get("image_url"),
            candidate_call_content[1].get("image_url"),
            evidence="corresponding annotated PNG commitment",
        )
        compare(
            f"{prefix}.recent_turn_boundaries",
            visible_steps(control_call_messages),
            visible_steps(candidate_call_messages),
            evidence="corresponding ordered observation-bearing turns",
        )
        compare(
            f"{prefix}.omission_notice",
            [
                _normalize_omission_notice(notice)
                for notice in _omission_notices(control_call_messages)
            ],
            [
                _normalize_omission_notice(notice)
                for notice in _omission_notices(candidate_call_messages)
            ],
            evidence="corresponding pointer-only omission behavior",
        )
        compare(
            f"{prefix}.candidate_tool_stability",
            candidate_tool_baseline,
            candidate_call.get("tools"),
            evidence="the candidate tool contract cannot drift between crossings",
        )
        compare(
            f"{prefix}.candidate_tool_contract",
            [],
            _candidate_tool_contract_errors(candidate_call.get("tools")),
            evidence="only one Python tool and one single-action return tool are exposed",
        )
        catalogue.append(
            {
                "path": f"{prefix}.actual_provider_payload",
                "classification": "measured_substrate_difference",
                "control": {"available": False, "crossings": []},
                "nooa": {
                    "available": bool(candidate_call.get("actual_provider_crossings", [])),
                    "crossings": candidate_call.get("actual_provider_crossings", []),
                },
                "evidence": "offline fixture or redacted actual provider crossing",
            }
        )

    visible_candidate = _host_supplied_model_surface(
        candidate_calls,
        workspace_contract=candidate_workspace_contract,
    )
    forbidden = detect_unmatched_model_affordances(visible_candidate)
    for affordance in forbidden:
        catalogue.append(
            {
                "path": f"forbidden_affordance.{affordance}",
                "classification": "unmatched_affordance",
                "control": "absent",
                "nooa": "present",
                "evidence": (
                    "system prompt exposes runtime terms or a forbidden cognitive "
                    "affordance"
                ),
            }
        )

    unmatched = [
        f"{entry['path']}: {entry['evidence']}; "
        f"control={entry['control']!r} nooa={entry['nooa']!r}"
        for entry in catalogue
        if entry["classification"] == "unmatched_affordance"
    ]
    report = {
        "schema": "0002-nooa-model-visible-parity-v2",
        "control_arm": "0002-R",
        "candidate_arm": "0002-NOOA",
        "fixture_coverage": {
            "candidate_actions": len(crossings),
            "control_actions": control_fixture["actions"],
            "successor_exercised": len(crossings) > 1,
            "omission_exercised": bool(_omission_notices(messages)),
            "canonical_request_exercised": isinstance(candidate_request_sha256, str),
            "actual_provider_exercised": bool(candidate_actual),
        },
        "catalogue": catalogue,
        "same_capability_different_transport": [
            entry
            for entry in catalogue
            if entry["classification"] == "same_capability_different_transport"
        ],
        "measured_substrate_differences": [
            entry
            for entry in catalogue
            if entry["classification"] == "measured_substrate_difference"
        ],
        "unmatched_affordances": unmatched,
        "canonical_requests": {
            "control": {
                "sha256": control_request_sha256,
                "encoding": "json-utf8-sort-keys-compact-v1",
                "content": _redact_data_images(control_request),
            },
            "nooa": {
                "sha256": candidate_request_sha256,
                "encoding": call.get("canonicalization"),
                "content": call.get("canonical_request", {"messages": messages, "tools": tools}),
            },
        },
    }
    (run_root / "parity-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
