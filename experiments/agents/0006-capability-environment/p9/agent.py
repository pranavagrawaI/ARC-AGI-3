"""One agent class from which both the 0002-R control and a 0006 treatment arm run.

The class is ``LlmReplAgent`` with two generic additions and no ARC-specific
cognition:

1. transcript eviction emits a pointer-only omission notice to the model;
2. an arm may enable named treatment capabilities from a fixed registry.

With no capabilities enabled this is the 0002-R control. The host supplies no
interpretation, relevance judgement, hypothesis management, or prediction
requirement.

A host-owned read-only observation archive was built here and removed. Its only
unique contribution was *authority* — a guarantee that the bytes the model reads
are the host's. Availability after eviction was never the gap: ``ObservationStore``
keeps every observation in the workspace for the whole run. Authority had no
measured demand and cost +816 prompt characters. See CONTRACT.md, "Deviation
from #9".
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Mapping

from llm_repl_agent.agent import PYTHON_TOOL, LlmReplAgent
from llm_repl_agent.observations import ModelObservation

from .capabilities import SENTINEL_NOTE, resolve_capabilities
from .conversation import AnnouncedConversation


OBSERVATIONS_GLOB = "observations/step-*.json"
# The step number is zero-padded to five digits. Nothing model-visible said so: the
# glob hides the width, and the only concrete filenames appear in recent_transitions,
# which is empty at step 0 and holds just the last eight steps. A model constructing a
# path guesses four digits and gets FileNotFoundError. Stated, and pinned to the real
# filenames by P9.1 so it cannot drift from ObservationStore.
OBSERVATION_STEP_DIGITS = 5
OBSERVATION_FILENAME_EXAMPLE = f"observations/step-{7:0{OBSERVATION_STEP_DIGITS}d}.json"


ARM_SYSTEM_PROMPT_ADDENDUM = """
If earlier turns leave this conversation you will be told which steps were dropped, and
nothing will be substituted for them. Every public observation of this run stays in
observations/step-NNNNN.json, where NNNNN is the step number padded with leading zeros to
five digits — step 7 is observations/step-00007.json. Steps whose turns are gone are still
there.

The interpreter runs isolated, so the workspace directory is not on sys.path. To import a
module you wrote, run sys.path.insert(0, ".") first. Names you define persist across python
calls until a timeout or crash restarts the interpreter, which is always reported to you;
files always persist.
"""


IMPORT_CONVENTION = (
    'The interpreter runs isolated: the workspace directory is not on sys.path. '
    'Run sys.path.insert(0, ".") before importing a module you wrote.\n'
)

PATH_CONVENTION = (
    "Observation files are observations/step-NNNNN.json with the step number padded to "
    f"five digits: step 7 is {OBSERVATION_FILENAME_EXAMPLE}.\n"
)

WORKSPACE_ADDENDUM = PATH_CONVENTION + IMPORT_CONVENTION


class SharedArmAgent(LlmReplAgent):
    """The shared arm implementation; capabilities select the treatment surface."""

    tools = (PYTHON_TOOL,)

    def __init__(
        self,
        *,
        arm_id: str,
        treatment_capabilities: tuple[str, ...] = (),
        **kwargs: Any,
    ) -> None:
        system_prompt = kwargs.get("system_prompt")
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("the shared arm requires an explicit system prompt")
        super().__init__(**kwargs)
        self.arm_id = arm_id
        self.capabilities = resolve_capabilities(tuple(treatment_capabilities))
        self.capability_names = tuple(capability.name for capability in self.capabilities)
        self.conversation = AnnouncedConversation(
            system_message={"role": "system", "content": system_prompt},
            max_turns=self.config.max_context_turns,
            observations_glob=OBSERVATIONS_GLOB,
        )
        with (self.workspace.root / "WORKSPACE.md").open("a", encoding="utf-8") as stream:
            stream.write(WORKSPACE_ADDENDUM)

    # -- model-visible surface ------------------------------------------------

    def _observation_message(
        self,
        current: ModelObservation,
        *,
        model_round: int,
        python_calls_this_action: int,
    ) -> dict[str, Any]:
        message = super()._observation_message(
            current,
            model_round=model_round,
            python_calls_this_action=python_calls_this_action,
        )
        payload = self._extend_payload(json.loads(message["content"][0]["text"]))
        message["content"][0]["text"] = json.dumps(payload, indent=2, sort_keys=True)
        return message

    def _extend_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload["environment_observation"]["observations_glob"] = OBSERVATIONS_GLOB
        if "sentinel_probe_note" in self.capability_names:
            payload["treatment"] = {"sentinel_probe_note": SENTINEL_NOTE}
        return payload

    def observation_contract(self) -> dict[str, Any]:
        """Probe the real message builder for its model-visible shape.

        The probe runs the production path against a stub observation, so the
        recorded contract cannot drift from what ``act`` actually sends.
        """

        stub = SimpleNamespace(
            metadata={},
            representations={},
            png_data_url="data:image/png;base64,",
        )
        message = self._observation_message(
            stub,  # type: ignore[arg-type]
            model_round=1,
            python_calls_this_action=0,
        )
        payload = json.loads(message["content"][0]["text"])
        return {
            "payload_keys": sorted(payload),
            "environment_observation_keys": sorted(payload["environment_observation"]),
            "deliberation_keys": sorted(payload["deliberation"]),
            "content_part_types": [part["type"] for part in message["content"]],
        }

    def tool_contract(self) -> list[Mapping[str, Any]]:
        return [json.loads(json.dumps(tool)) for tool in self.tools]
