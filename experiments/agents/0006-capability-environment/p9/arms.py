"""The single factory both arms are built from, and the manifest it commits to.

``build_arm`` is the only construction path. The 0002-R control and any 0006
treatment differ solely by ``ArmSpec.treatment_capabilities``; every other
runtime field is shared by construction, not by convention.

``arm_manifest`` reads the *constructed* arm — the live system message, the tool
list the agent will send, a probe of the real observation-message builder, the
actual workspace mounts — rather than the spec that requested it. A difference
that never reaches the manifest is a gap in the manifest, so the manifest also
carries the source digests of every participating module.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from typing import Any

from llm_repl_agent.agent import SYSTEM_PROMPT, LlmAgentConfig
from llm_repl_agent.client import ChatClient
from llm_repl_agent.observations import ColorPermutation
from llm_repl_agent.process_log import ProcessLog
from llm_repl_agent.workspace import MODEL_OUTPUT_CHAR_LIMIT, PythonWorkspace

from .agent import ARM_SYSTEM_PROMPT_ADDENDUM, SharedArmAgent
from .capabilities import capability_prompt_addendum
from .conversation import omission_notice_template_sha256
from .failures import failure_taxonomy
from .transport import RetryingChatClient


MANIFEST_SCHEMA = "p9-arm-manifest-v1"
CONTROL_ARM_ID = "0002-R"
TREATMENT_ARM_ID = "0006-provisional"

ARM_SYSTEM_PROMPT = SYSTEM_PROMPT.rstrip() + "\n" + ARM_SYSTEM_PROMPT_ADDENDUM

REPO = Path(__file__).resolve().parents[4]
P9_ROOT = Path(__file__).resolve().parent
BASELINE_PACKAGE = REPO / "experiments" / "agents" / "0002-llm-repl-agent" / "llm_repl_agent"


@dataclass(frozen=True)
class ArmSpec:
    """Everything that distinguishes one arm from another. Only the last field may."""

    arm_id: str
    system_prompt: str = ARM_SYSTEM_PROMPT
    max_context_turns: int = 6
    max_model_rounds_per_action: int = 10
    max_decision_repairs: int = 2
    max_model_calls: int | None = 100
    max_reported_tokens: int | None = 250_000
    drop_stale_python_context: bool = False
    workspace_timeout_seconds: float = 20.0
    max_actions: int = 500
    transport_max_retries: int = 2
    transport_backoff_seconds: float = 0.0
    color_permutation_seed: int | None = None
    treatment_capabilities: tuple[str, ...] = ()

    def agent_config(self) -> LlmAgentConfig:
        return LlmAgentConfig(
            max_context_turns=self.max_context_turns,
            max_model_rounds_per_action=self.max_model_rounds_per_action,
            max_decision_repairs=self.max_decision_repairs,
            max_model_calls=self.max_model_calls,
            max_reported_tokens=self.max_reported_tokens,
            drop_stale_python_context=self.drop_stale_python_context,
        )


def control_spec(**overrides: Any) -> ArmSpec:
    return replace(ArmSpec(arm_id=CONTROL_ARM_ID), **overrides)


def treatment_spec(
    control: ArmSpec,
    *,
    arm_id: str = TREATMENT_ARM_ID,
    treatment_capabilities: tuple[str, ...] = ("sentinel_probe_note",),
) -> ArmSpec:
    """Derive the treatment from the control so shared fields cannot drift apart."""

    return replace(
        control,
        arm_id=arm_id,
        treatment_capabilities=tuple(sorted(treatment_capabilities)),
    )


@dataclass
class Arm:
    spec: ArmSpec
    run_dir: Path
    process_log: ProcessLog
    workspace: PythonWorkspace
    inner_client: ChatClient
    client: RetryingChatClient
    agent: SharedArmAgent
    _closed: bool = field(default=False, repr=False)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.workspace.close()
        finally:
            self.process_log.close()

    def __enter__(self) -> "Arm":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def build_arm(spec: ArmSpec, *, run_dir: Path, inner_client: ChatClient) -> Arm:
    """Construct one arm. Both arms in a pair must come through this function."""

    run_dir.mkdir(parents=True, exist_ok=True)
    process_log = ProcessLog(run_dir / "process.jsonl")
    workspace = PythonWorkspace(timeout_seconds=spec.workspace_timeout_seconds)
    client = RetryingChatClient(
        inner_client,
        process_log=process_log,
        max_retries=spec.transport_max_retries,
        backoff_seconds=spec.transport_backoff_seconds,
    )
    colors = (
        ColorPermutation.from_seed(spec.color_permutation_seed)
        if spec.color_permutation_seed is not None
        else ColorPermutation.identity()
    )
    agent_class = SharedArmAgent
    if "learning_loop" in spec.treatment_capabilities:
        # Keep treatment cognition outside the frozen control import closure. The
        # shared factory selects it only for the declared capability.
        module = import_module(f"{__package__}.learning_loop_agent")
        agent_class = module.LearningLoopAgent
    agent = agent_class(
        arm_id=spec.arm_id,
        treatment_capabilities=spec.treatment_capabilities,
        client=client,
        workspace=workspace,
        process_log=process_log,
        system_prompt=spec.system_prompt,
        color_permutation=colors,
        config=spec.agent_config(),
    )
    process_log.write(
        "arm_built",
        arm_id=spec.arm_id,
        treatment_capabilities=list(spec.treatment_capabilities),
    )
    return Arm(
        spec=spec,
        run_dir=run_dir,
        process_log=process_log,
        workspace=workspace,
        inner_client=inner_client,
        client=client,
        agent=agent,
    )


def participating_sources() -> dict[str, str]:
    """Digest every module either arm executes, so identical arms prove identical code."""

    paths = set(P9_ROOT.glob("*.py"))
    paths.update(BASELINE_PACKAGE.glob("*.py"))
    paths.update(
        (REPO / "experiments" / "harness" / name)
        for name in ("runner.py", "types.py", "trace.py")
    )
    return {
        path.relative_to(REPO).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def arm_manifest(arm: Arm) -> dict[str, Any]:
    """Describe every model↔harness crossing of a constructed arm."""

    agent = arm.agent
    system_prompt = str(agent.conversation.system_message["content"])
    # The base contract and the capability's own text are hashed apart, so a
    # capability can document itself without authorising a difference in the
    # contract underneath it. A smuggled edit to the base still fails.
    addendum = capability_prompt_addendum(arm.spec.treatment_capabilities)
    base_prompt = system_prompt[: len(system_prompt) - len(addendum)] if addendum else system_prompt
    return {
        "schema": MANIFEST_SCHEMA,
        "arm_id": arm.spec.arm_id,
        "model_visible": {
            "system_prompt": {
                "sha256": sha256(base_prompt.encode("utf-8")).hexdigest(),
                "capability_addendum_sha256": sha256(addendum.encode("utf-8")).hexdigest(),
                "chars": len(base_prompt),
            },
            "tools": agent.tool_contract(),
            "observation_message": agent.observation_contract(),
            "deliberation_budgets": {
                "max_model_rounds_per_action": agent.config.max_model_rounds_per_action,
                "max_decision_repairs": agent.config.max_decision_repairs,
                "action_required_on_final_round": True,
            },
            "workspace": {
                "writable_root": "/workspace",
                "reserved_paths": sorted(
                    path.name for path in arm.workspace.root.iterdir()
                ),
                "contract_sha256": sha256(
                    (arm.workspace.root / "WORKSPACE.md").read_bytes()
                ).hexdigest(),
                "provided_helper_sha256": sha256(
                    agent.provided_helper.encode("utf-8")
                ).hexdigest(),
                "python_output_char_limit": MODEL_OUTPUT_CHAR_LIMIT,
                "timeout_seconds": arm.workspace.timeout_seconds,
                "allowed_distributions": sorted(
                    arm.workspace.runtime_info()["allowed_distributions"]
                ),
            },
            "context_policy": {
                "max_turns": agent.config.max_context_turns,
                "python_context_policy": (
                    "drop_stale"
                    if agent.config.drop_stale_python_context
                    else "full"
                ),
                "omission_notice_template_sha256": omission_notice_template_sha256(),
                "host_authored_summary": False,
            },
            "failure_taxonomy": failure_taxonomy(),
            "decision_contract": (
                agent.decision_contract()
                if hasattr(agent, "decision_contract")
                else {
                    "parser": "llm_repl_agent.agent.parse_decision_message",
                    "fields": ["action", "data", "reason"],
                    "legality_checked_against": "observation.available_actions",
                }
            ),
            "treatment_capabilities": [
                {
                    "name": capability.name,
                    "summary": capability.summary,
                    "manifest_paths": list(capability.manifest_paths),
                    "request_payload_paths": list(capability.request_payload_paths),
                }
                for capability in agent.capabilities
            ],
        },
        "host_only": {
            "spec": {
                "arm_id": arm.spec.arm_id,
                "max_actions": arm.spec.max_actions,
                "max_model_calls": arm.spec.max_model_calls,
                "max_reported_tokens": arm.spec.max_reported_tokens,
                "color_permutation_seed": arm.spec.color_permutation_seed,
                "treatment_capabilities": list(arm.spec.treatment_capabilities),
            },
            "transport_retry_policy": arm.client.policy(),
            # The registry as a whole is deliberately absent. Listing every declared
            # capability made an arm's manifest depend on capabilities it does not
            # enable, so registering a treatment moved the control's manifest digest
            # and forced a re-pin — a moving baseline, and the exact failure the
            # freeze exists to catch. Scoped to what this arm enables, the field is
            # spec.treatment_capabilities above. What else was declared that run is
            # recoverable from capabilities.py, which runtime_digest() covers.
            "source_sha256": participating_sources(),
        },
    }
