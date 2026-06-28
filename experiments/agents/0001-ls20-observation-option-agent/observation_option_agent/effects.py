"""Conditional visual-effect evidence with a non-contact baseline."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .events import Event
from .interactions import InteractionEvent
from .tracking import TrackUpdate


@dataclass(frozen=True)
class ContactDescriptor:
    value: int
    area: int
    shape_id: str


ContactSignature = tuple[ContactDescriptor, ...]


@dataclass(frozen=True)
class EffectKey:
    target_object_id: int
    target_value: int
    kind: str


@dataclass(frozen=True)
class EffectTransition:
    effect: EffectKey
    before_state: str
    after_state: str


@dataclass(frozen=True)
class EffectEvidence:
    level: int
    contacts: tuple[ContactDescriptor, ...]
    present_object_ids: frozenset[int]
    effects: frozenset[EffectKey]
    transitions: tuple[EffectTransition, ...] = ()
    step: int = 0


CompositeState = tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class StateTransition:
    step: int
    before: CompositeState
    after: CompositeState


@dataclass(frozen=True)
class AttributeStateModel:
    observed_state_count: int
    deterministic: bool
    continuous: bool
    closed_cycle: bool
    cycle_length: int | None
    current_state: CompositeState | None
    transitions: tuple[StateTransition, ...]


@dataclass(frozen=True)
class TargetEvidence:
    object_id: int
    value: int
    baseline_support: int
    baseline_opportunities: int
    baseline_rate: float
    rate_excess: float


@dataclass(frozen=True)
class Affordance:
    level: int
    contacts: ContactSignature
    effect_kind: str
    target_object_ids: tuple[int, ...]
    targets: tuple[TargetEvidence, ...]
    contact_support: int
    contact_opportunities: int
    contact_rate: float
    state_model: AttributeStateModel


def extract_effect_evidence(
    event: Event,
    tracks: TrackUpdate,
    interaction: InteractionEvent,
) -> EffectEvidence | None:
    """Compress one interaction into neutral contact/effect symbols."""

    if event.level_delta or event.full_reset:
        return None
    controlled_ids = set(interaction.controlled_object_ids)
    before_by_id = {match.object_id: match.before for match in tracks.matches}
    before_by_id.update(
        (item.object_id, item.detection) for item in tracks.disappeared
    )
    before_by_id.update(
        (item.object_id, item.detection) for item in tracks.hidden
    )
    contacts = tuple(sorted({
        ContactDescriptor(
            value=before_by_id[item.object_id].value,
            area=before_by_id[item.object_id].area,
            shape_id=before_by_id[item.object_id].shape_id,
        )
        for item in interaction.contact_candidates
        if item.object_id in before_by_id
    }, key=lambda item: (item.value, item.area, item.shape_id)))
    present_object_ids = frozenset(
        object_id
        for object_id in before_by_id
        if object_id not in controlled_ids
    )
    effects: set[EffectKey] = set()
    transitions: list[EffectTransition] = []
    for delta in event.object_deltas:
        if delta.object_id in controlled_ids:
            continue
        before = before_by_id.get(delta.object_id)
        after = next(
            (
                match.after
                for match in tracks.matches
                if match.object_id == delta.object_id
            ),
            None,
        )
        if before is None or after is None:
            continue
        state_before = f"{before.value}:{before.shape_id}"
        state_after = f"{after.value}:{after.shape_id}"
        if delta.value_changed:
            effect = EffectKey(delta.object_id, delta.value_before, "value_change")
            effects.add(effect)
            transitions.append(EffectTransition(effect, state_before, state_after))
        if delta.shape_changed or delta.area_delta:
            effect = EffectKey(
                delta.object_id,
                delta.value_before,
                delta.shape_transform,
            )
            effects.add(effect)
            transitions.append(EffectTransition(effect, state_before, state_after))
    return EffectEvidence(
        level=event.levels_before,
        contacts=contacts,
        present_object_ids=present_object_ids,
        effects=frozenset(effects),
        transitions=tuple(transitions),
        step=event.step,
    )


class ConditionalEffectMemory:
    """Keep effects elevated above their autonomous non-contact rate."""

    def __init__(
        self,
        *,
        minimum_contact_support: int = 3,
        minimum_baseline_opportunities: int = 5,
        minimum_rate_excess: float = 0.5,
    ) -> None:
        self.minimum_contact_support = minimum_contact_support
        self.minimum_baseline_opportunities = minimum_baseline_opportunities
        self.minimum_rate_excess = minimum_rate_excess
        self._contact_opportunities: Counter[tuple[int, ContactSignature]] = Counter()
        self._contact_effects: Counter[tuple[int, ContactSignature, EffectKey]] = Counter()
        self._baseline_opportunities: Counter[tuple[int, int]] = Counter()
        self._baseline_effects: Counter[tuple[int, EffectKey]] = Counter()
        self._contact_records: list[EffectEvidence] = []

    def observe(self, evidence: EffectEvidence) -> None:
        if evidence.contacts:
            self._contact_records.append(evidence)
            self._contact_opportunities[(evidence.level, evidence.contacts)] += 1
            for effect in evidence.effects:
                self._contact_effects[(evidence.level, evidence.contacts, effect)] += 1
            return

        for object_id in evidence.present_object_ids:
            self._baseline_opportunities[(evidence.level, object_id)] += 1
        for effect in evidence.effects:
            self._baseline_effects[(evidence.level, effect)] += 1

    def summaries(self) -> list[dict[str, object]]:
        summaries = []
        for (level, contact_signature, effect), support in self._contact_effects.items():
            opportunities = self._contact_opportunities[(level, contact_signature)]
            baseline_opportunities = self._baseline_opportunities[
                (level, effect.target_object_id)
            ]
            if support < self.minimum_contact_support:
                continue
            if baseline_opportunities < self.minimum_baseline_opportunities:
                continue
            contact_rate = support / opportunities
            baseline_support = self._baseline_effects[(level, effect)]
            baseline_rate = baseline_support / baseline_opportunities
            excess = contact_rate - baseline_rate
            if excess < self.minimum_rate_excess:
                continue
            summaries.append({
                "level": level,
                "contact": [
                    {
                        "value": contact.value,
                        "area": contact.area,
                        "shape_id": contact.shape_id,
                    }
                    for contact in contact_signature
                ],
                "effect": {
                    "target_object_id": effect.target_object_id,
                    "target_value": effect.target_value,
                    "kind": effect.kind,
                },
                "contact_support": support,
                "contact_opportunities": opportunities,
                "contact_rate": round(contact_rate, 4),
                "baseline_support": baseline_support,
                "baseline_opportunities": baseline_opportunities,
                "baseline_rate": round(baseline_rate, 4),
                "rate_excess": round(excess, 4),
            })
        summaries.sort(
            key=lambda item: (
                item["rate_excess"],
                item["contact_support"],
                item["contact_rate"],
            ),
            reverse=True,
        )
        return summaries

    def affordances(self) -> tuple[Affordance, ...]:
        """Return accepted effects as queryable neutral relations.

        Targets are grouped only when the same composite contact repeatedly
        produces the same effect kind at the same conditional rate.
        """

        grouped: dict[
            tuple[int, ContactSignature, str, int, int],
            list[TargetEvidence],
        ] = {}
        for summary in self.summaries():
            effect = summary["effect"]
            contacts = tuple(
                ContactDescriptor(
                    value=item["value"],
                    area=item["area"],
                    shape_id=item["shape_id"],
                )
                for item in summary["contact"]
            )
            key = (
                summary["level"],
                contacts,
                effect["kind"],
                summary["contact_support"],
                summary["contact_opportunities"],
            )
            grouped.setdefault(key, []).append(TargetEvidence(
                object_id=effect["target_object_id"],
                value=effect["target_value"],
                baseline_support=summary["baseline_support"],
                baseline_opportunities=summary["baseline_opportunities"],
                baseline_rate=summary["baseline_rate"],
                rate_excess=summary["rate_excess"],
            ))

        affordances = []
        for (
            level,
            contacts,
            effect_kind,
            support,
            opportunities,
        ), targets in grouped.items():
            ordered_targets = tuple(sorted(targets, key=lambda item: item.object_id))
            object_ids = tuple(item.object_id for item in ordered_targets)
            affordances.append(Affordance(
                level=level,
                contacts=contacts,
                effect_kind=effect_kind,
                target_object_ids=object_ids,
                targets=ordered_targets,
                contact_support=support,
                contact_opportunities=opportunities,
                contact_rate=support / opportunities,
                state_model=self._state_model(
                    level=level,
                    contacts=contacts,
                    effect_kind=effect_kind,
                    target_object_ids=object_ids,
                ),
            ))
        affordances.sort(
            key=lambda item: (
                item.effect_kind != "unclassified_form_change",
                item.contact_support,
                len(item.target_object_ids),
            ),
            reverse=True,
        )
        return tuple(affordances)

    def _state_model(
        self,
        *,
        level: int,
        contacts: ContactSignature,
        effect_kind: str,
        target_object_ids: tuple[int, ...],
    ) -> AttributeStateModel:
        transitions = []
        for record in self._contact_records:
            if record.level != level or record.contacts != contacts:
                continue
            by_target = {
                transition.effect.target_object_id: transition
                for transition in record.transitions
                if transition.effect.kind == effect_kind
            }
            if not all(object_id in by_target for object_id in target_object_ids):
                continue
            transitions.append(StateTransition(
                step=record.step,
                before=tuple(
                    (object_id, by_target[object_id].before_state)
                    for object_id in target_object_ids
                ),
                after=tuple(
                    (object_id, by_target[object_id].after_state)
                    for object_id in target_object_ids
                ),
            ))

        states = {
            state
            for transition in transitions
            for state in (transition.before, transition.after)
        }
        successors: dict[CompositeState, set[CompositeState]] = {}
        predecessors: dict[CompositeState, set[CompositeState]] = {}
        for transition in transitions:
            successors.setdefault(transition.before, set()).add(transition.after)
            predecessors.setdefault(transition.after, set()).add(transition.before)
        deterministic = bool(transitions) and all(
            len(items) == 1 for items in successors.values()
        )
        continuous = bool(transitions) and all(
            left.after == right.before
            for left, right in zip(transitions, transitions[1:])
        )
        closed_cycle = (
            deterministic
            and continuous
            and bool(states)
            and set(successors) == states
            and set(predecessors) == states
            and all(len(items) == 1 for items in predecessors.values())
        )
        return AttributeStateModel(
            observed_state_count=len(states),
            deterministic=deterministic,
            continuous=continuous,
            closed_cycle=closed_cycle,
            cycle_length=len(states) if closed_cycle else None,
            current_state=transitions[-1].after if transitions else None,
            transitions=tuple(transitions),
        )
