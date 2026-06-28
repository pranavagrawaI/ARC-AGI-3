"""Generic prequential transition prediction over neutral event records."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Any
from collections.abc import Callable


ContactSignature = tuple[tuple[int, int, str], ...]


def contact_signature_from_record(record: dict[str, Any]) -> ContactSignature:
    interaction = record.get("interaction") or {}
    candidates = interaction.get("contact_candidates") or ()
    return tuple(sorted(
        (
            int(item["value"]),
            int(item["object_area"]),
            str(item["shape_id"]),
        )
        for item in candidates
    ))


def transition_symbols_from_record(record: dict[str, Any]) -> frozenset[str]:
    symbols: set[str] = set()
    interaction = record.get("interaction") or {}
    if interaction.get("blocked"):
        symbols.add("blocked")

    level_delta = int(record.get("levels_after", 0)) - int(record.get("levels_before", 0))
    if level_delta:
        symbols.add(f"level_delta:{level_delta:+d}")
    if record.get("full_reset"):
        symbols.add("full_reset")

    for delta in record.get("object_deltas", ()):
        translation = tuple(delta.get("translation", (0, 0)))
        moved = translation != (0, 0)
        transform = str(delta.get("shape_transform", "same"))
        if delta.get("shape_changed") and transform != "same":
            symbols.add(f"transform:{transform}")
        if delta.get("value_changed"):
            symbols.add("value_change")
        if int(delta.get("area_delta", 0)):
            symbols.add("area_change")
        if moved:
            symbols.add(f"motion:{translation[0]}:{translation[1]}")

    if record.get("created_object_ids"):
        symbols.add("created")
    if record.get("disappeared_object_ids"):
        symbols.add("disappeared")
    if record.get("reappeared_object_ids"):
        symbols.add("reappeared")
    return frozenset(symbols)


def contact_effect_symbols_from_record(record: dict[str, Any]) -> frozenset[str]:
    interaction = record.get("interaction") or {}
    contact_candidates = interaction.get("contact_candidates") or ()
    if not contact_candidates:
        return frozenset()

    changed_other = set(interaction.get("changed_other_object_ids") or ())
    symbols: set[str] = set()
    for delta in record.get("object_deltas", ()):
        if int(delta.get("object_id", -1)) not in changed_other:
            continue
        transform = str(delta.get("shape_transform", "same"))
        if delta.get("shape_changed") and transform != "same":
            symbols.add(f"effect_transform:{transform}")
        if delta.get("value_changed"):
            symbols.add("effect_value_change")
        if int(delta.get("area_delta", 0)):
            symbols.add("effect_area_change")
    if interaction.get("blocked"):
        symbols.add("contact_blocked")
    if record.get("levels_after", 0) != record.get("levels_before", 0):
        symbols.add("contact_level_delta")
    return frozenset(symbols)


@dataclass(frozen=True)
class TransitionPrediction:
    probabilities: dict[str, float]
    source: str
    observations: int

    def probability(self, symbol: str) -> float:
        return self.probabilities.get(symbol, 0.0)


@dataclass(frozen=True)
class PrequentialResult:
    examples: int
    non_empty_examples: int
    mean_observed_probability: float
    top1_accuracy: float
    average_symbol_log_loss: float


@dataclass(frozen=True)
class RelationHypothesis:
    contact_signature: ContactSignature
    support: int
    top_effects: tuple[tuple[str, float, float], ...]


class PrequentialTransitionLearner:
    """Predict generic event symbols from admissible historical records.

    The model is intentionally small: it stores empirical symbol frequencies for
    action+contact, action-only, and global contexts. Prediction always happens
    before the current record is observed by callers using ``score_records``.
    """

    def __init__(
        self,
        *,
        smoothing: float = 1.0,
        contexts: tuple[str, ...] = ("action_contact", "action", "global"),
        symbol_extractor: Callable[[dict[str, Any]], frozenset[str]] = (
            transition_symbols_from_record
        ),
    ) -> None:
        self.smoothing = smoothing
        self.contexts = contexts
        self.symbol_extractor = symbol_extractor
        self._counts: dict[tuple[Any, ...], Counter[str]] = defaultdict(Counter)
        self._observations: Counter[tuple[Any, ...]] = Counter()
        self._vocabulary: set[str] = set()

    def predict(self, record: dict[str, Any]) -> TransitionPrediction:
        available = [
            (source, key)
            for source, key in self._context_keys(record)
            if self._observations[key]
        ]
        if available:
            return TransitionPrediction(
                probabilities=self._pooled_probabilities(tuple(key for _source, key in available)),
                source=available[0][0],
                observations=sum(self._observations[key] for _source, key in available),
            )
        return TransitionPrediction({}, "untrained", 0)

    def predict_hard_backoff(self, record: dict[str, Any]) -> TransitionPrediction:
        for source, key in self._context_keys(record):
            if self._observations[key]:
                return TransitionPrediction(
                    probabilities=self._probabilities(key),
                    source=source,
                    observations=self._observations[key],
                )
        return TransitionPrediction({}, "untrained", 0)

    def observe(self, record: dict[str, Any]) -> None:
        symbols = self.symbol_extractor(record)
        self._vocabulary.update(symbols)
        for _source, key in self._context_keys(record):
            self._observations[key] += 1
            self._counts[key].update(symbols)

    def score_records(self, records: list[dict[str, Any]]) -> PrequentialResult:
        observed_probability_sum = 0.0
        top1_hits = 0
        symbol_log_loss = 0.0
        symbol_count = 0
        non_empty = 0
        for record in records:
            prediction = self.predict(record)
            actual = self.symbol_extractor(record)
            if actual:
                non_empty += 1
                observed_probability_sum += sum(
                    prediction.probability(symbol) for symbol in actual
                ) / len(actual)
                top_symbol = max(
                    prediction.probabilities,
                    key=prediction.probabilities.get,
                    default=None,
                )
                if top_symbol in actual:
                    top1_hits += 1
                for symbol in actual:
                    probability = max(prediction.probability(symbol), 1e-9)
                    symbol_log_loss -= math.log(probability)
                    symbol_count += 1
            self.observe(record)

        return PrequentialResult(
            examples=len(records),
            non_empty_examples=non_empty,
            mean_observed_probability=(
                observed_probability_sum / non_empty if non_empty else 0.0
            ),
            top1_accuracy=top1_hits / non_empty if non_empty else 0.0,
            average_symbol_log_loss=symbol_log_loss / symbol_count if symbol_count else 0.0,
        )

    def _context_keys(self, record: dict[str, Any]) -> tuple[tuple[str, tuple[Any, ...]], ...]:
        action = int(record["action"])
        contact_signature = contact_signature_from_record(record)
        keys = (
            ("action_contact", ("action_contact", action, contact_signature)),
            ("action", ("action", action)),
            ("global", ("global",)),
        )
        return tuple((source, key) for source, key in keys if source in self.contexts)

    def _probabilities(self, key: tuple[Any, ...]) -> dict[str, float]:
        return self._pooled_probabilities((key,))

    def _pooled_probabilities(self, keys: tuple[tuple[Any, ...], ...]) -> dict[str, float]:
        if not self._vocabulary:
            return {}
        observations = sum(self._observations[key] for key in keys)
        denominator = (
            observations
            + self.smoothing * len(self._vocabulary)
        )
        return {
            symbol: (
                sum(self._counts[key][symbol] for key in keys) + self.smoothing
            ) / denominator
            for symbol in sorted(self._vocabulary)
        }


class RelationEffectLearner:
    """Learn effect distributions for neutral contact signatures."""

    def __init__(self, *, smoothing: float = 1.0) -> None:
        self._learner = PrequentialTransitionLearner(
            contexts=("action_contact", "global"),
            smoothing=smoothing,
            symbol_extractor=contact_effect_symbols_from_record,
        )
        self._contact_counts: Counter[ContactSignature] = Counter()
        self._contact_effects: dict[ContactSignature, Counter[str]] = defaultdict(Counter)
        self._global_contact_count = 0
        self._global_effects: Counter[str] = Counter()

    def predict(self, record: dict[str, Any]) -> TransitionPrediction:
        return self._learner.predict(record)

    def observe(self, record: dict[str, Any]) -> None:
        effects = contact_effect_symbols_from_record(record)
        contact_signature = contact_signature_from_record(record)
        self._learner.observe(record)
        if not contact_signature:
            return
        self._contact_counts[contact_signature] += 1
        self._contact_effects[contact_signature].update(effects)
        self._global_contact_count += 1
        self._global_effects.update(effects)

    def score_records(self, records: list[dict[str, Any]]) -> PrequentialResult:
        observed_probability_sum = 0.0
        top1_hits = 0
        symbol_log_loss = 0.0
        symbol_count = 0
        non_empty = 0
        for record in records:
            prediction = self.predict(record)
            actual = contact_effect_symbols_from_record(record)
            if actual:
                non_empty += 1
                observed_probability_sum += sum(
                    prediction.probability(symbol) for symbol in actual
                ) / len(actual)
                top_symbol = max(
                    prediction.probabilities,
                    key=prediction.probabilities.get,
                    default=None,
                )
                if top_symbol in actual:
                    top1_hits += 1
                for symbol in actual:
                    probability = max(prediction.probability(symbol), 1e-9)
                    symbol_log_loss -= math.log(probability)
                    symbol_count += 1
            self.observe(record)

        return PrequentialResult(
            examples=len(records),
            non_empty_examples=non_empty,
            mean_observed_probability=(
                observed_probability_sum / non_empty if non_empty else 0.0
            ),
            top1_accuracy=top1_hits / non_empty if non_empty else 0.0,
            average_symbol_log_loss=symbol_log_loss / symbol_count if symbol_count else 0.0,
        )

    def hypotheses(
        self,
        *,
        minimum_support: int = 2,
        top_k: int = 3,
    ) -> tuple[RelationHypothesis, ...]:
        output = []
        for contact_signature, support in self._contact_counts.items():
            if support < minimum_support:
                continue
            effects = self._contact_effects[contact_signature]
            if not effects:
                continue
            scored_effects = []
            for effect, count in effects.items():
                rate = count / support
                global_rate = (
                    self._global_effects[effect] / self._global_contact_count
                    if self._global_contact_count
                    else 0.0
                )
                lift = rate / global_rate if global_rate else float("inf")
                scored_effects.append((effect, rate, lift))
            scored_effects.sort(
                key=lambda item: (item[2], item[1], item[0]),
                reverse=True,
            )
            top_effects = tuple(
                (effect, round(rate, 4), round(lift, 4))
                for effect, rate, lift in scored_effects[:top_k]
            )
            output.append(RelationHypothesis(
                contact_signature=contact_signature,
                support=support,
                top_effects=top_effects,
            ))
        output.sort(
            key=lambda item: (
                item.support,
                item.top_effects[0][1] if item.top_effects else 0.0,
            ),
            reverse=True,
        )
        return tuple(output)
