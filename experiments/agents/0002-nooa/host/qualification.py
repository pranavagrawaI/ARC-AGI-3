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
from typing import Any


@dataclass(frozen=True)
class PairedRunSpec:
    """The settings that must not drift independently between the two bases."""

    max_context_turns: int
    max_model_rounds_per_action: int
    max_decision_repairs: int
    max_model_calls: int
    max_reported_tokens: int
    workspace_timeout_seconds: float
    max_actions: int
    transport_max_retries: int

    @classmethod
    def initial(cls) -> "PairedRunSpec":
        return cls(
            max_context_turns=6,
            max_model_rounds_per_action=10,
            max_decision_repairs=2,
            max_model_calls=100,
            max_reported_tokens=250_000,
            workspace_timeout_seconds=20.0,
            max_actions=500,
            transport_max_retries=2,
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

    def as_record(self) -> dict[str, int | float]:
        """Return the canonical IPC/audit projection of the shared settings."""

        return asdict(self)


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
        first_lines = (control_root / destination).read_text(encoding="utf-8").splitlines()[:2]
        if not any("SPDX-FileCopyrightText" in line for line in first_lines):
            mismatches.append(f"derived source lacks copyright notice: {destination}")

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
        for term in ("nooa", "agent instance", "self.", "context_manager", "event_manager")
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
