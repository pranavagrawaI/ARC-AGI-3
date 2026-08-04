"""Authorized treatment capabilities and the manifest paths each one may move.

A treatment arm is the control arm plus a set of named capabilities. Each entry
declares, in advance, exactly which normalized manifest paths and which
model-visible request-payload paths it is allowed to change. The arm verifier
rejects any difference outside those declarations.

``sentinel_probe_note`` exists only to prove the verifier discriminates. It adds
one inert field and carries no evidence, belief, prediction, or routing
semantics.
"""

from __future__ import annotations

from dataclasses import dataclass


SENTINEL_NOTE = (
    "Sentinel treatment capability. It carries no information about this run and "
    "requires nothing of you."
)


@dataclass(frozen=True)
class TreatmentCapability:
    name: str
    summary: str
    manifest_paths: tuple[str, ...]
    request_payload_paths: tuple[str, ...]
    # Documented in the prompt or not at all. A live pair shipped an operator that
    # was described only inside the helper's docstring, neither arm ever opened the
    # file, and it was used zero times. A capability nobody is told about cannot be
    # measured. The addendum is hashed separately from the base prompt so a
    # capability may move its own text and never the contract underneath it.
    prompt_addendum: str = ""


TREATMENT_CAPABILITIES: dict[str, TreatmentCapability] = {
    "learning_loop": TreatmentCapability(
        name="learning_loop",
        summary=(
            "Preserves an arbitrary model-authored working state, returns only exact "
            "model-selected transition records, separates immediate mechanism predictions "
            "from bounded objective-progress claims, and reports both mechanical results. "
            "The host does not create, rank, or revise hypotheses."
        ),
        manifest_paths=(
            "model_visible.decision_contract",
            "model_visible.observation_message.payload_keys",
            "model_visible.treatment_capabilities",
            "model_visible.workspace.reserved_paths",
        ),
        request_payload_paths=(
            "messages.*.content.*.observation_payload.capability_packet",
        ),
        prompt_addendum=(
            "The learning_loop capability adds capability_packet to each observation. "
            "It echoes your prior working_state, the mechanical result of your preceding "
            "mechanism prediction, active_objective, last_objective_result, and exact "
            "transition records for history_steps you selected on your preceding action. "
            "Observation pixels remain at the packet's file paths. "
            "For every action, reason must be an object with exactly: working_state (any "
            "JSON object you own), test (a brief mechanism test), prediction (1-5 immediate "
            "public mechanism assertions), objective, history_steps (0-4 already recorded "
            "integer steps to retrieve next turn), and reconciliation. An assertion is "
            "{observable:'changed_pixels'|'levels_completed',op:'=='|'!='|'<'|'<='|'>'|'>=',value:nonnegative integer}, "
            "{observable:'state',op:'=='|'!=',value:string}, or "
            "{observable:'cell',x:column,y:row,op:'=='|'!=',value:integer}; cell reads the "
            "public final layer. When active_objective is null, objective must be "
            "{claim:string,progress:{observable:'levels_completed',op:'>',value:CURRENT_LEVELS_COMPLETED},horizon:1..8}. "
            "This is a bounded game-progress claim, not another local visual prediction. "
            "While active_objective exists, objective must be null so its horizon cannot be "
            "reset. reconciliation must be null unless the preceding mechanism prediction "
            "mismatched or last_objective_result is falsified; after either, it must explain "
            "your response. A falsified objective requires a new model-authored objective. "
            "The host preserves and checks this material but never decides what it means."
        ),
    ),
    "connected_components": TreatmentCapability(
        name="connected_components",
        summary=(
            "Adds frames.components() — connected runs of equal-valued cells. "
            "Registered rather than shipped in the control: 4 of 4 recorded scipy "
            "imports were scipy.ndimage.label, so the demand is real, but adjacency "
            "labelling is a contested theory of objectness that this repository's own "
            "findings call insufficient (LR 0014, 0017, 0019). It is here to be "
            "measured against a control that lacks it."
        ),
        manifest_paths=(
            "model_visible.workspace.provided_helper_sha256",
            "model_visible.treatment_capabilities",
        ),
        # The first live pair documented this operator only inside the helper
        # docstring and got zero adoption: neither arm ever opened frames.py. The
        # prompt now carries an addendum, hashed separately from the shared base so
        # a smuggled prompt difference is still caught. No observation payload
        # differs, so request_payload_paths stays empty.
        request_payload_paths=(),
        prompt_addendum=(
            "frames.components(step) is also available: it returns connected runs of "
            "equal-valued cells as {value, size, bbox, cells}, largest first. It groups "
            "by adjacency only — it says nothing about what any group is, does not track "
            "anything across steps, and will split a multi-coloured shape."
        ),
    ),
    "sentinel_probe_note": TreatmentCapability(
        name="sentinel_probe_note",
        summary="Adds one inert model-visible field; changes no other surface.",
        manifest_paths=(
            "model_visible.observation_message.payload_keys",
            "model_visible.treatment_capabilities",
        ),
        request_payload_paths=("messages.*.content.*.observation_payload.treatment",),
    ),
}


def resolve_capabilities(names: tuple[str, ...]) -> tuple[TreatmentCapability, ...]:
    unknown = sorted(set(names) - set(TREATMENT_CAPABILITIES))
    if unknown:
        raise ValueError(f"unknown treatment capabilities: {unknown}")
    return tuple(TREATMENT_CAPABILITIES[name] for name in sorted(set(names)))


def capability_prompt_addendum(names: tuple[str, ...]) -> str:
    """The text appended to the shared contract by the enabled capabilities."""

    parts = [c.prompt_addendum for c in resolve_capabilities(names) if c.prompt_addendum]
    return ("\n\n" + "\n".join(parts)) if parts else ""


def authorized_manifest_paths(names: tuple[str, ...]) -> frozenset[str]:
    paths = {"arm_id", "host_only.spec.arm_id", "host_only.spec.treatment_capabilities"}
    for capability in resolve_capabilities(names):
        paths.update(capability.manifest_paths)
        if capability.prompt_addendum:
            paths.add("model_visible.system_prompt.capability_addendum_sha256")
    return frozenset(paths)


def authorized_request_payload_paths(names: tuple[str, ...]) -> frozenset[str]:
    paths: set[str] = set()
    for capability in resolve_capabilities(names):
        paths.update(capability.request_payload_paths)
    return frozenset(paths)
