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

from base64 import b64encode
import json
from types import SimpleNamespace
from typing import Any, Mapping

from llm_repl_agent.agent import PYTHON_TOOL, LlmReplAgent
from llm_repl_agent.observations import ModelObservation, observation_metadata
from llm_repl_agent.workspace import MODEL_RUNTIME_DISTRIBUTIONS

from .annotated_frame import annotate, annotation_descriptor
from .capabilities import SENTINEL_NOTE, capability_prompt_addendum, resolve_capabilities
from .frame_tools import FRAME_TOOLS_FILE, FRAME_TOOLS_USAGE, frame_tools_source
from .conversation import AnnouncedConversation


OBSERVATIONS_GLOB = "observations/step-*.json"
# The step number is zero-padded to five digits. Nothing model-visible said so: the
# glob hides the width, and the only concrete filenames appear in recent_transitions,
# which is empty at step 0 and holds just the last eight steps. A model constructing a
# path guesses four digits and gets FileNotFoundError. Stated, and pinned to the real
# filenames by P9.1 so it cannot drift from ObservationStore.
OBSERVATION_STEP_DIGITS = 5
OBSERVATION_FILENAME_EXAMPLE = f"observations/step-{7:0{OBSERVATION_STEP_DIGITS}d}.json"
# ObservationStore writes a rendered PNG beside every observation JSON. Nothing said
# so, and a live model wrote a defensive fallback to current_frame.png with the
# comment "Since I don't have the files explicitly". It never failed - it just
# declined to use a capability it had, which no friction log can see.
OBSERVATION_IMAGE_EXAMPLE = f"observations/step-{7:0{OBSERVATION_STEP_DIGITS}d}.png"
# Three live runs: the width is always right now, the separator is not. Both residual
# failures were composition errors in an f-string - 'step00000.json' and
# 'step_00036.json' - never a wrong width. Stating the pattern still leaves the model
# assembling "step-" next to a format spec, so state the expression it will actually
# write. P9.1 evaluates this string and requires the result to exist.
# Play-test findings, all verified against payloads before being acted on.
# The frozen system prompt says the workspace "starts empty on every independent
# run". Five entries exist before the model's first call. A probe model noticed and
# filed it as a doc inaccuracy; it is a contradiction in the model-visible contract.
# A model-authored file the harness echoes back verbatim every request, so the model's
# own conclusions survive transcript eviction. The host never reads, selects from, or
# summarises it - it is a dumb echo, which is what keeps it generic. Production coding
# agents carry the same thing. A play-along probe named its absence the single worst
# thing about the harness, having lost its action semantics and map to an eviction and
# had to rebuild them from files.
# One image, not seven. A request carried one per retained turn, six of them
# superseded frames; the current frame is the only one showing the present, and the
# earlier ones remain on disk. Explicitly replaced rather than silently dropped.
SUPERSEDED_FRAME_NOTE = (
    "[the frame image for this earlier step is omitted; every frame remains on disk "
    "as observations/step-NNNNN.png, and the current frame is shown with the newest "
    "observation]"
)

NOTES_FILE = "NOTES.md"
NOTES_CHAR_LIMIT = 4000

RESERVED_WORKSPACE_ENTRIES = (
    "NOTES.md",
    "frames.py",
    "WORKSPACE.md",
    "current_frame.png",
    "current_observation.json",
    "observations",
    "public_history.jsonl",
)
# The layer count is not stable. Across the recorded corpus 40 of 648 observations
# had more than one layer (2, 5, 6, 13, 15, 22), while 404 of 591 tool calls indexed
# pixels[0]. pixels_to_png renders pixels[-1], so on those steps the model parses a
# stale layer while the image it is shown displays a different one: no error, no log
# entry, wrong state. A play-along probe caught it at step 9 of its own run.
CURRENT_LAYER_INDEX = -1

# self._history[-8:] in the frozen agent. The field is named "recent" and the cap is
# nowhere stated; a probe model found it by trial at step 9.
RECENT_TRANSITIONS_LIMIT = 8
OBSERVATION_PATH_EXPRESSION = (
    'f"observations/step-{step:0' + str(OBSERVATION_STEP_DIGITS) + 'd}.json"'
)

# Derived from the frozen allowlist, so the statement cannot drift from the sandbox.
# 7 of 47 recorded tool failures were the model importing something that is not there.
AVAILABLE_IMPORTS = tuple(sorted(roots[0] for roots in MODEL_RUNTIME_DISTRIBUTIONS.values()))


def _observation_schema_keys() -> tuple[str, ...]:
    """The top-level keys of an observation file, derived from the frozen transform.

    4 of 47 recorded tool failures were the model guessing a key that is not there.
    The contract described the file as "flat objects with top-level public metadata"
    without ever saying which keys that meant.
    """

    import numpy as np

    from experiments.harness import Observation

    probe = Observation(
        pixels=np.zeros((1, 1, 1), dtype=int),
        available_actions=(1,),
        levels_completed=0,
        state="NOT_FINISHED",
        full_reset=False,
    )
    return tuple(sorted({*observation_metadata(probe), "pixels", "step"}))


OBSERVATION_SCHEMA_KEYS = _observation_schema_keys()


ARM_SYSTEM_PROMPT_ADDENDUM = f"""
If earlier turns leave this conversation you will be told which steps were dropped, and
nothing will be substituted for them. The workspace is new for this run but not empty: it
already holds {", ".join(RESERVED_WORKSPACE_ENTRIES)}. recent_transitions in each request
carries the most recent {RECENT_TRANSITIONS_LIMIT} only, while public_history.jsonl carries
all of them, and the digest fields are opaque state identifiers rather than a hash you can
recompute. frames.py in your workspace loads and compares observations for you —
`import sys; sys.path.insert(0, "."); import frames` then frames.diff(a, b) — and it reads
the current layer correctly. NOTES.md in your workspace is echoed back to you in full at the top of every
request and survives dropped turns; the harness never reads or changes it. pixels is
indexed [layer][row][column], the layer count changes between steps,
and the current state is always the last layer: read pixels[-1], never pixels[0]. The
rendered image shows that same last layer. Every public observation of this run stays in
observations/step-NNNNN.json, where NNNNN is the step number padded with leading zeros to
five digits — step 7 is observations/step-00007.json. Steps whose turns are gone are still
there, and each step is also rendered as an image beside it — step 7 is
{OBSERVATION_FILENAME_EXAMPLE} and {OBSERVATION_IMAGE_EXAMPLE}; in Python the path is
{OBSERVATION_PATH_EXPRESSION}. Each JSON file has exactly
these top-level keys: {", ".join(OBSERVATION_SCHEMA_KEYS)}.

The sandbox has the Python standard library plus {" and ".join(AVAILABLE_IMPORTS)} only.
Nothing else is installed and there is no network, so an import of anything else will fail.

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
    f"five digits: step 7 is {OBSERVATION_FILENAME_EXAMPLE}. In Python that path is "
    f"{OBSERVATION_PATH_EXPRESSION}. Every step is also rendered "
    f"as an image at the same name: {OBSERVATION_IMAGE_EXAMPLE}.\n"
)

SCHEMA_CONVENTION = (
    "Each observation file has exactly these top-level keys: "
    + ", ".join(OBSERVATION_SCHEMA_KEYS)
    + ".\n"
)

WORKSPACE_CONTENTS_CONVENTION = (
    "The workspace is new for this run but not empty: it already contains "
    + ", ".join(RESERVED_WORKSPACE_ENTRIES)
    + ". Everything else in it is yours.\n"
)

FRAME_TOOLS_CONVENTION = (
    f"{FRAME_TOOLS_FILE} is provided for loading and comparing observations: "
    f"`{FRAME_TOOLS_USAGE}` then frames.steps(), frames.load(n), frames.grid(n), "
    "frames.diff(a, b), frames.where(n, value). grid() returns the current layer "
    "for you. Read it, edit it, replace it — it is an ordinary file in your "
    "workspace.\n"
)

IMAGE_CONVENTION = (
    "The image shows the current frame with grid coordinates on the top and left "
    "margins: x is the column along the top, y is the row down the left, both 0-based "
    "and exactly the numbers an action takes. Read coordinates off the rulers rather "
    "than estimating them. Only the current frame is shown as an image; earlier frames "
    "stay on disk as observations/step-NNNNN.png.\n"
)

NOTES_CONVENTION = (
    f"{NOTES_FILE} is yours. Whatever you put in it is echoed back to you in full at the "
    f"top of every request, so anything you write there survives conversation turns being "
    f"dropped. The harness never reads, edits, summarises or acts on it. It is capped at "
    f"{NOTES_CHAR_LIMIT} characters; past that you are shown the last "
    f"{NOTES_CHAR_LIMIT} and told it was truncated.\n"
)

LAYER_CONVENTION = (
    "pixels is indexed [layer][row][column] and the number of layers changes between "
    "steps. The current state is always the LAST layer: use pixels[-1], not pixels[0]. "
    "The rendered image always shows that same last layer, and representations."
    "rendered_image.source_grid_layer names which one it was.\n"
)

HISTORY_CONVENTION = (
    f"recent_transitions in each request holds the most recent "
    f"{RECENT_TRANSITIONS_LIMIT} transitions only; public_history.jsonl holds all of "
    "them. The before_digest and after_digest fields are opaque identifiers for "
    "comparing states, not a hash you can recompute from the files.\n"
)

RUNTIME_CONVENTION = (
    "The sandbox has the Python standard library plus "
    + " and ".join(AVAILABLE_IMPORTS)
    + " only; nothing else is installed and there is no network.\n"
)

WORKSPACE_ADDENDUM = (
    WORKSPACE_CONTENTS_CONVENTION
    + FRAME_TOOLS_CONVENTION
    + IMAGE_CONVENTION
    + NOTES_CONVENTION
    + LAYER_CONVENTION
    + PATH_CONVENTION
    + SCHEMA_CONVENTION
    + HISTORY_CONVENTION
    + RUNTIME_CONVENTION
    + IMPORT_CONVENTION
)


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
            system_message={
                "role": "system",
                "content": system_prompt + capability_prompt_addendum(self.capability_names),
            },
            max_turns=self.config.max_context_turns,
            observations_glob=OBSERVATIONS_GLOB,
            superseded_frame_note=SUPERSEDED_FRAME_NOTE,
        )
        with (self.workspace.root / "WORKSPACE.md").open("a", encoding="utf-8") as stream:
            stream.write(WORKSPACE_ADDENDUM)
        # Every observation payload advertises full_history_file from step 0, but
        # ObservationStore creates it only on the first transition. A probe model
        # checked and found it missing, then had to discover it appears later.
        # Advertising a file that does not exist is the same class of defect as
        # not advertising one that does.
        (self.workspace.root / "public_history.jsonl").touch()
        (self.workspace.root / NOTES_FILE).touch()
        # Both recorded models rewrote this every few turns and paid a deliberation
        # round each time. It is measurement over public files, not perception.
        self.provided_helper = frame_tools_source(
            components="connected_components" in self.capability_names
        )
        (self.workspace.root / FRAME_TOOLS_FILE).write_text(
            self.provided_helper, encoding="utf-8"
        )

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
        pixels = getattr(current, "pixels", None)
        if pixels is not None:
            annotated = annotate(pixels)
            for part in message["content"]:
                if part.get("type") == "image_url":
                    part["image_url"] = {
                        "url": "data:image/png;base64," + b64encode(annotated).decode("ascii")
                    }
        return message

    # The frozen payload carries these both at top level and inside
    # environment_observation, labelled "backward-compatible top-level mirrors for
    # older prompts/tests". 0002-R has no older prompt to be compatible with, and a
    # play-along probe filed it as waste: "the request envelope is enormously
    # redundant... scrolling past thousands of duplicated tokens to reach the one new
    # thing got old by the third turn". Measured at 47% of the observation payload.
    DUPLICATED_TOP_LEVEL_KEYS = (
        "current_metadata",
        "current_observation_file",
        "full_history_file",
        "recent_transitions",
        "representations",
    )

    def _read_notes(self) -> dict[str, Any]:
        """Echo the model's own notes back verbatim, truncation stated."""

        path = self.workspace.root / NOTES_FILE
        text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        truncated = len(text) > NOTES_CHAR_LIMIT
        return {
            "file": NOTES_FILE,
            "chars": len(text),
            "limit": NOTES_CHAR_LIMIT,
            "truncated": truncated,
            # Keep the tail: the file is append-only in practice, so the newest
            # conclusions are the ones worth preserving.
            "showing": "last {} characters".format(NOTES_CHAR_LIMIT) if truncated else "all",
            "content": text[-NOTES_CHAR_LIMIT:] if truncated else text,
        }

    def _extend_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload["notes"] = self._read_notes()
        representations = payload["environment_observation"].get("representations")
        if isinstance(representations, dict) and "rendered_image" in representations:
            representations["rendered_image"]["annotation"] = annotation_descriptor()
        for key in self.DUPLICATED_TOP_LEVEL_KEYS:
            # Only drop the mirror; the canonical copy lives under
            # environment_observation and must survive.
            if key in payload and key in payload["environment_observation"]:
                payload.pop(key)
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
