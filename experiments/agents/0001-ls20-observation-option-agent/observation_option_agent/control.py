"""Online induction of which neutral visual group is action-controlled."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from .events import Delta, Event


@dataclass(frozen=True)
class ControlEstimate:
    level: int
    object_ids: tuple[int, ...]
    support: int
    distinct_actions: int
    consistency: float
    confidence: float
    source: str
    action_model: dict[int, Delta]
    discovered_step: int
    actions_into_level: int


class ControlledEntityBelief:
    """Earn a controlled role from immediate action-conditioned motion.

    Object IDs are level-local. Only an action-to-displacement model transfers
    across public level boundaries; the visual group must be grounded again.
    """

    def __init__(self, *, minimum_support: int = 6, prior_action_matches: int = 2) -> None:
        self.minimum_support = minimum_support
        self.prior_action_matches = prior_action_matches
        self.level = 0
        self._level_start_step = 0
        self.active: ControlEstimate | None = None
        self.discoveries: list[ControlEstimate] = []
        self._local: dict[tuple[int, ...], dict[int, Counter[Delta]]] = defaultdict(
            lambda: defaultdict(Counter)
        )
        self._transferred: dict[int, Counter[Delta]] = defaultdict(Counter)

    def observe(self, event: Event) -> ControlEstimate | None:
        if event.levels_before != self.level:
            self._start_level(event.levels_before, start_step=event.step - 1)

        for group in event.motion_groups:
            self._local[group.object_ids][event.action][group.translation] += 1

        if self.active is not None:
            self._update_transferred_from_active(event)
        else:
            candidate = self._best_candidate(event.step)
            if candidate is not None:
                self.active = candidate
                self.discoveries.append(candidate)
                self._seed_transferred(candidate.object_ids)

        estimate = self.active
        if event.level_delta > 0:
            self._start_level(event.levels_after, start_step=event.step)
        return estimate

    def expected_delta(self, action: int) -> Delta | None:
        if self.active is not None and action in self.active.action_model:
            return self.active.action_model[action]
        counter = self._transferred.get(action)
        return counter.most_common(1)[0][0] if counter else None

    def _start_level(self, level: int, *, start_step: int) -> None:
        self.level = level
        self._level_start_step = start_step
        self.active = None
        self._local = defaultdict(lambda: defaultdict(Counter))

    def _best_candidate(self, step: int) -> ControlEstimate | None:
        candidates = []
        transferred_model = {
            action: counter.most_common(1)[0][0]
            for action, counter in self._transferred.items()
            if counter
        }
        for object_ids, by_action in self._local.items():
            action_model = {}
            support = correct = 0
            for action, counter in by_action.items():
                delta, count = counter.most_common(1)[0]
                action_model[action] = delta
                support += sum(counter.values())
                correct += count
            distinct_deltas = len(set(action_model.values()))
            consistency = correct / support
            prior_matches = {
                action
                for action, delta in action_model.items()
                if transferred_model.get(action) == delta
            }
            prior_ready = (
                len(prior_matches) >= self.prior_action_matches
                and len(set(action_model[action] for action in prior_matches)) >= 2
            )
            evidence_ready = (
                support >= self.minimum_support
                and len(action_model) >= 2
                and distinct_deltas >= 2
            )
            if not prior_ready and not evidence_ready:
                continue
            source = "transferred_action_model" if prior_ready else "local_action_evidence"
            dependence = distinct_deltas / len(action_model)
            confidence = consistency * dependence * min(1.0, support / self.minimum_support)
            estimate = ControlEstimate(
                level=self.level,
                object_ids=object_ids,
                support=support,
                distinct_actions=len(action_model),
                consistency=round(consistency, 4),
                confidence=round(confidence, 4),
                source=source,
                action_model=dict(sorted(action_model.items())),
                discovered_step=step,
                actions_into_level=step - self._level_start_step,
            )
            candidates.append(estimate)
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda item: (
                item.source == "transferred_action_model",
                item.confidence,
                item.distinct_actions,
                item.support,
            ),
        )

    def _seed_transferred(self, object_ids: tuple[int, ...]) -> None:
        for action, counter in self._local[object_ids].items():
            self._transferred[action].update(counter)

    def _update_transferred_from_active(self, event: Event) -> None:
        for group in event.motion_groups:
            if group.object_ids == self.active.object_ids:
                self._transferred[event.action][group.translation] += 1
                break
