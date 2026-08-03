"""The fixed commitment-preserving controller state machine.

It answers one question:

    Does an eight-field state block, plus rules that force a provisional
    abstraction to survive an action, admit competent play while refusing the
    drift that sank Gemma?

The controller supplies no conclusions and no actions. It never says where the
player is, what a cell contains, or which action to take. It only decides
whether a proposed state update is a legal continuation of the previous one.
Every rejection names a rule, so a rejection is legible to the model.

It is active only under the ``rules_controller`` knowledge condition. Every
other condition and both arms are unaffected.

`prototypes/commitment_controller/probe.py` replays recorded trajectories
through this exact module offline, with no provider call. Run it before
changing a rule: `--replay luna` must stay at 28/28 admitted, because that is
the only recorded trajectory that has completed LS20 level 1.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

FIELDS = (
    "PLAYER_BINDING",
    "LOCAL_OBJECTIVE",
    "ROUTE1",
    "CURRENT_APPROX_POSITION",
    "LAST_ACTION",
    "LAST_OUTCOME",
    "NEXT_STEP",
    "CONTRADICTION",
)
OUTCOMES = ("moved", "blocked", "special_event", "uncertain")
UNKNOWN = "unknown"
# Only these three may be left unknown. An unknown NEXT_STEP is a refusal to
# act, and an unknown LAST_OUTCOME is what "uncertain" is for.
MAY_BE_UNKNOWN = ("PLAYER_BINDING", "CURRENT_APPROX_POSITION", "ROUTE1")


class Rejected(Exception):
    """One violated rule, named."""

    def __init__(self, rule: str, detail: str) -> None:
        super().__init__(f"{rule}: {detail}")
        self.rule = rule
        self.detail = detail


@dataclass(frozen=True)
class State:
    """The whole persistent abstraction. Deliberately too small to hide in."""

    PLAYER_BINDING: str = UNKNOWN
    LOCAL_OBJECTIVE: str = ""
    ROUTE1: tuple[int, ...] = ()
    CURRENT_APPROX_POSITION: str = UNKNOWN
    LAST_ACTION: int | None = None
    LAST_OUTCOME: str | None = None
    NEXT_STEP: int | None = None
    CONTRADICTION: str = ""
    # Bookkeeping the model never writes.
    settled: bool = False
    uncertain_streak: int = 0
    history: tuple[str, ...] = field(default=(), repr=False)

    def render(self) -> str:
        """The exact text seeded into NOTES.md and echoed every model round."""
        route = ",".join(str(a) for a in self.ROUTE1) if self.ROUTE1 else UNKNOWN
        lines = [
            f"PLAYER_BINDING: {self.PLAYER_BINDING}",
            f"LOCAL_OBJECTIVE: {self.LOCAL_OBJECTIVE or UNKNOWN}",
            f"ROUTE1: {route}",
            f"CURRENT_APPROX_POSITION: {self.CURRENT_APPROX_POSITION}",
            f"LAST_ACTION: {self.LAST_ACTION if self.LAST_ACTION is not None else 'none'}",
            f"LAST_OUTCOME: {self.LAST_OUTCOME or 'none'}",
            f"NEXT_STEP: {self.NEXT_STEP if self.NEXT_STEP is not None else UNKNOWN}",
            f"CONTRADICTION: {self.CONTRADICTION or 'none'}",
        ]
        return "\n".join(lines)


def classify(changed_cells: int, *, moved: int, quiet: int) -> str:
    """Mechanical ground truth for scoring the model's own classification.

    This is a scoring instrument, not part of the controller. The live host
    never shows the model a label; it shows the neutral ``changed_cells`` count
    the existing evidence operator already produces and requires the model to
    name the outcome itself. Handing over the label would hand over the one
    semantic mapping the probe is trying to measure.
    """

    if changed_cells <= quiet:
        return "blocked"
    if changed_cells == moved:
        return "moved"
    return "special_event"


def _is_local_repair(old: tuple[int, ...], new: tuple[int, ...], *, detour: int) -> bool:
    """A repaired route still aims where the old one aimed.

    ROUTE1 is the *remaining* plan, so the ordinary turn consumes its head. The
    first prototype run got this wrong: it demanded that the new route end in a
    tail of the old one, which forbids both consumption and planning further
    ahead, and so rejected 15 of Luna's 28 winning turns. The rule that survives
    contact with real play is weaker and states the actual intent -- whatever
    remained of the old plan must still be in the new one, reachable within a
    bounded detour:

        new == <=detour repair steps ++ old[1:] ++ any extension

    Consuming a step, prepending a detour around a wall, and planning further
    ahead are all legal. Dropping the remainder is not; that is replacement, and
    replacement needs a contradiction.
    """

    rest = old[1:]
    if not rest:
        return True
    for start in range(0, min(detour, len(new)) + 1):
        if new[start : start + len(rest)] == rest:
            return True
    return False


class CommitmentController:
    """Enforces commitment preservation across one action per turn."""

    def __init__(self, *, detour: int = 3, uncertain_limit: int = 2) -> None:
        self.state = State()
        self.detour = detour
        self.uncertain_limit = uncertain_limit
        self.log: list[dict[str, object]] = []
        # R6 was drafted as a hard rule and the Luna replay demoted it. Three of
        # her 28 winning turns replace ROUTE1 outright after a move revealed the
        # corridor's real shape -- competent re-planning, not reinterpretation.
        # Enforcing it would force the model either to declare a contradiction
        # it does not believe or to walk a route it knows is wrong. What has to
        # survive an action is the binding and the objective; the action list is
        # the cheap part. So route replacement is counted and reported, and the
        # hard rules stay on identity.
        self.route_replacements = 0

    # -- the seven rules ---------------------------------------------------
    def _check(self, old: State, new: dict[str, object], legal: tuple[int, ...]) -> None:
        unknown_keys = set(new) - set(FIELDS)
        if unknown_keys:
            raise Rejected("R0_FIELDS", f"unknown fields {sorted(unknown_keys)}")

        binding = str(new.get("PLAYER_BINDING", old.PLAYER_BINDING))
        objective = str(new.get("LOCAL_OBJECTIVE", old.LOCAL_OBJECTIVE))
        route = tuple(new.get("ROUTE1", old.ROUTE1))  # type: ignore[arg-type]
        outcome = new.get("LAST_OUTCOME", old.LAST_OUTCOME)
        contradiction = str(new.get("CONTRADICTION", "") or "")
        last_action = new.get("LAST_ACTION", old.LAST_ACTION)
        nxt = new.get("NEXT_STEP", None)

        if not all(
            isinstance(step, int) and not isinstance(step, bool) and step in legal
            for step in route
        ):
            raise Rejected(
                "R0_FIELDS", f"ROUTE1 must contain only legal actions from {legal}"
            )
        if outcome is not None and outcome not in OUTCOMES:
            raise Rejected("R0_FIELDS", f"LAST_OUTCOME must be one of {OUTCOMES}")
        for name in FIELDS:
            value = new.get(name)
            if value == UNKNOWN and name not in MAY_BE_UNKNOWN:
                raise Rejected("R4_UNKNOWN", f"{name} may not be unknown")

        # R7: exactly one legal action per turn. This is the one-action bound.
        if isinstance(nxt, bool) or not isinstance(nxt, int) or nxt not in legal:
            raise Rejected("R7_ONE_ACTION", f"NEXT_STEP must be one legal action of {legal}")
        first_turn = not old.history
        if isinstance(last_action, bool):
            raise Rejected("R0_FIELDS", "LAST_ACTION must be an integer action or none")
        if first_turn and last_action is not None:
            raise Rejected("R0_FIELDS", "LAST_ACTION must be none on the first turn")
        if not first_turn and last_action != old.NEXT_STEP:
            raise Rejected(
                "R0_FIELDS",
                f"LAST_ACTION must equal the prior NEXT_STEP {old.NEXT_STEP!r}",
            )

        binding_changed = old.settled and binding != old.PLAYER_BINDING
        objective_changed = bool(old.LOCAL_OBJECTIVE) and objective != old.LOCAL_OBJECTIVE
        route_replaced = bool(old.ROUTE1) and not _is_local_repair(
            old.ROUTE1, route, detour=self.detour
        )

        # R1: a settled binding survives unless this very turn contradicts it.
        # R5 is its one exemption -- a qualitatively novel event licenses focused
        # semantic inspection without the model having to claim it was wrong.
        if binding_changed and not contradiction and outcome != "special_event":
            raise Rejected(
                "R1_BINDING_PERSISTS",
                "PLAYER_BINDING changed with no CONTRADICTION and no special_event",
            )

        # R2 and R3 say an ordinary outcome does not *by itself* license
        # reinterpretation. They must still yield to a named contradiction, or
        # the most natural way to discover a mis-binding in this game -- "I moved
        # and the thing I called the player stayed put" -- could never be acted
        # on, and the model's only escape would be to stop classifying its own
        # transitions. A test caught this; the first draft omitted the exemption
        # from R2 and R3 while granting it in R1, which made a contradiction
        # discovered by movement permanently unusable.
        if not contradiction:
            # R2: ordinary movement is not evidence about what anything is.
            if outcome == "moved" and (binding_changed or objective_changed):
                raise Rejected(
                    "R2_NO_RESCAN_ON_MOVE",
                    "a successful ordinary move may not revise binding or "
                    "objective; name a CONTRADICTION if one really broke",
                )

            # R3: a wall is a fact about one cell, not about the world.
            if outcome == "blocked" and (
                binding_changed or objective_changed or route == old.ROUTE1
            ):
                raise Rejected(
                    "R3_BLOCKED_IS_LOCAL",
                    "a blocked move must reroute without revising binding or "
                    "objective; name a CONTRADICTION if one really broke",
                )

        # R6 is measured, not enforced -- see the note on ``route_replacements``.
        # R8: a contradiction licenses exactly one reinterpretation, then closes.
        if contradiction and old.CONTRADICTION:
            raise Rejected(
                "R8_CONTRADICTION_CONSUMED",
                "the previous CONTRADICTION was never cleared",
            )

        # Standing uncertainty is the failure mode itself. Two turns of it and
        # the model must either commit to a binding or declare a contradiction.
        if outcome == "uncertain" and old.uncertain_streak >= self.uncertain_limit:
            raise Rejected(
                "R9_NO_STANDING_DOUBT",
                f"{old.uncertain_streak + 1} consecutive uncertain outcomes; "
                "name a CONTRADICTION or classify the transition",
            )

    def propose(
        self, update: dict[str, object], *, legal: tuple[int, ...], force: bool = False
    ) -> State:
        """Accept or reject one state update. Raises Rejected with a rule name.

        ``force`` applies the update even when a rule rejects it, and is only for
        replay. Without it a single rejection freezes the state and every later
        turn is scored against a stale route, which turns one disagreement into
        twenty-five.
        """

        old = self.state
        try:
            self._check(old, update, legal)
        except Rejected as rejection:
            self.log.append(
                {"accepted": False, "forced": force, "rule": rejection.rule}
            )
            if not force:
                raise
        route = tuple(update.get("ROUTE1", old.ROUTE1))  # type: ignore[arg-type]
        if old.ROUTE1 and not _is_local_repair(old.ROUTE1, route, detour=self.detour):
            self.route_replacements += 1
        outcome = update.get("LAST_OUTCOME", old.LAST_OUTCOME)
        binding = str(update.get("PLAYER_BINDING", old.PLAYER_BINDING))
        new = replace(
            old,
            PLAYER_BINDING=binding,
            LOCAL_OBJECTIVE=str(update.get("LOCAL_OBJECTIVE", old.LOCAL_OBJECTIVE)),
            ROUTE1=tuple(update.get("ROUTE1", old.ROUTE1)),  # type: ignore[arg-type]
            CURRENT_APPROX_POSITION=str(
                update.get("CURRENT_APPROX_POSITION", old.CURRENT_APPROX_POSITION)
            ),
            LAST_ACTION=update.get("LAST_ACTION", old.LAST_ACTION),  # type: ignore[arg-type]
            LAST_OUTCOME=outcome,  # type: ignore[arg-type]
            NEXT_STEP=update["NEXT_STEP"],  # type: ignore[index]
            CONTRADICTION=str(update.get("CONTRADICTION", "") or ""),
            settled=old.settled or binding != UNKNOWN,
            uncertain_streak=old.uncertain_streak + 1 if outcome == "uncertain" else 0,
            history=old.history + (binding,),
        )
        self.state = new
        self.log.append({"accepted": True, "state": new.render()})
        return new

    # -- the live entry point ----------------------------------------------
    def admit_working_state(
        self,
        working_state: object,
        *,
        action: int,
        legal_actions: Sequence[int],
    ) -> State:
        """Check one commitment's ``working_state`` as the next controller turn.

        The model authors every value. This only decides whether the new block
        is a legal continuation of the old one, and that the action it is about
        to take is the one the block committed to.
        """

        if not isinstance(working_state, Mapping):
            raise Rejected("R0_FIELDS", "working_state must be an object")
        missing = [name for name in FIELDS if name not in working_state]
        if missing:
            raise Rejected(
                "R0_FIELDS",
                f"working_state must contain exactly the {len(FIELDS)} controller "
                f"fields; missing {missing}",
            )
        update = dict(working_state)
        route = update.get("ROUTE1")
        if isinstance(route, str):
            # The block is authored as text in NOTES.md, so accept the same
            # comma-separated form here rather than making the model remember
            # which of the two surfaces wants a list.
            stripped = route.strip()
            if stripped in {"", UNKNOWN, "none"}:
                update["ROUTE1"] = ()
            else:
                parts = [part.strip() for part in stripped.split(",")]
                if not parts or any(not part.lstrip("-").isdigit() for part in parts):
                    raise Rejected(
                        "R0_FIELDS", "ROUTE1 must be comma-separated integer actions"
                    )
                update["ROUTE1"] = tuple(int(part) for part in parts)
        elif isinstance(route, list):
            update["ROUTE1"] = tuple(route)
        for name in ("LAST_ACTION", "NEXT_STEP"):
            value = update.get(name)
            if isinstance(value, str):
                text = value.strip()
                if name == "LAST_ACTION" and text in {"", "none"}:
                    update[name] = None
                elif text.lstrip("-").isdigit():
                    update[name] = int(text)
                else:
                    raise Rejected(
                        "R0_FIELDS", f"{name} must be an integer action or none"
                    )
        if update.get("LAST_OUTCOME") in {"", "none", None}:
            update["LAST_OUTCOME"] = self.state.LAST_OUTCOME
        if update.get("NEXT_STEP") != action:
            raise Rejected(
                "R7_ONE_ACTION",
                f"NEXT_STEP {update.get('NEXT_STEP')!r} is not the committed "
                f"action {action}",
            )
        return self.propose(update, legal=tuple(legal_actions))

    # -- the measurements the experiment reports ---------------------------
    def metrics(self) -> dict[str, int]:
        settled = [b for b in self.state.history if b != UNKNOWN]
        reversals = sum(
            1 for a, b in zip(settled, settled[1:]) if a != b
        )
        return {
            "identity_reversals": reversals,
            "route_replacements": self.route_replacements,
            "turns": len(self.state.history),
            "rejections": sum(1 for entry in self.log if not entry.get("accepted")),
        }
