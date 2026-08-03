"""PROTOTYPE driver. One command, no spend, no provider call.

    uv run python experiments/agents/0006-nooa/prototypes/commitment_controller/probe.py

Three modes:

  --replay luna    push the recorded 28-action winning trajectory through the
                   controller. Every turn must be admitted; a rejection means
                   the rules would have blocked competent play.
  --replay drift   push hand-built violations of each rule through it. Every
                   turn must be rejected, by the named rule.
  (default)        interactive. Type a field update, watch the whole state.

The point of the interactive mode is the cases that are hard to argue about on
paper: a blocked move followed by a reroute followed by a changer event, with a
contradiction opened and never closed.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

# Import the module that actually runs live, so this replay is a test of the
# production controller rather than of a copy that can drift from it.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "host"))

from commitment_controller import (  # noqa: E402
    FIELDS,
    CommitmentController as Controller,
    Rejected,
    classify,
)

HERE = pathlib.Path(__file__).parent
LEGAL = (1, 2, 3, 4)
MOVED_CELLS = 52
QUIET_CELLS = 2


def _banner(text: str) -> None:
    print(f"\n\033[1m{text}\033[0m\n" + "-" * len(text))


def _contains(haystack: tuple[int, ...], needle: tuple[int, ...]) -> bool:
    return any(
        haystack[i : i + len(needle)] == needle
        for i in range(len(haystack) - len(needle) + 1)
    )


def _show(controller: Controller, note: str = "") -> None:
    print(controller.state.render())
    if note:
        print(f"  -> {note}")


def replay_luna() -> int:
    """Does the controller admit play that actually won?"""

    turns = json.loads((HERE / "fixtures/luna_win.json").read_text())
    controller = Controller()
    rejections: list[str] = []
    misclassified = 0
    route_survived = 0
    local_repairs = 0
    global_reinterps = 0

    objective = "reach shape changer"
    binding = "unknown"
    route: tuple[int, ...] = ()

    for i, turn in enumerate(turns):
        # The outcome a turn can react to is the *previous* transition. Reading
        # the current turn's own changed_cells would let the reconstruction know
        # the future, which is exactly the mistake that made the first run look
        # like an R1 failure at turn 20.
        prior = (
            classify(turns[i - 1]["changed_cells"], moved=MOVED_CELLS, quiet=QUIET_CELLS)
            if i
            else None
        )
        prose = json.dumps(turn["working_state"]).lower()
        # Luna's prose is her own classification; score it against the diff.
        claimed = "blocked" if "blocked" in prose else prior
        if i and claimed != prior:
            misclassified += 1

        # A special_event is the one thing that licenses re-inspecting identity.
        if prior == "special_event":
            binding = "player = two-colour block, shape advanced"
            objective = "reach goal"
        elif binding == "unknown":
            binding = "player = two-colour block, original shape"

        remaining = route[1:]
        if prior == "blocked":
            # Local repair: a detour step in front of whatever still remains.
            route = (turn["action"],) + remaining
            local_repairs += 1
        elif remaining and remaining[0] == turn["action"]:
            route = remaining  # the plan survived; consume its head
            route_survived += 1
        else:
            # Re-plan from here. Legal only if it still contains the remainder.
            route = tuple(t["action"] for t in turns[i:][:4])
            if remaining and _contains(route, remaining):
                route_survived += 1

        update = {
            "PLAYER_BINDING": binding,
            "LOCAL_OBJECTIVE": objective,
            "ROUTE1": route,
            "CURRENT_APPROX_POSITION": str(turn["working_state"].get("route", ""))[:60],
            "LAST_ACTION": turns[i - 1]["action"] if i else None,
            "LAST_OUTCOME": prior,
            "NEXT_STEP": turn["action"],
            "CONTRADICTION": "",
        }
        legal = tuple(turn["available_actions"]) or LEGAL
        try:
            controller.propose(update, legal=legal)
        except Rejected as exc:
            rejections.append(f"turn {turn['action_id']}: {exc}")
            global_reinterps += 1
            controller.propose(update, legal=legal, force=True)

    _banner("replay: luna, 28 actions, level 1 completed")
    print(f"turns admitted            {len(turns) - len(rejections)}/{len(turns)}")
    print(f"identity reversals        {controller.metrics()['identity_reversals']}")
    print(f"route1 survived turns     {route_survived}/{len(turns) - 1}")
    print(f"blocked repaired locally  {local_repairs}")
    print(f"route replacements        {controller.metrics()['route_replacements']}")
    print(f"outcome misclassified     {misclassified}")
    for line in rejections:
        print(f"  REJECTED {line}")
    _show(controller, "final state")
    if rejections:
        print("\nVERDICT: the rules block competent play. Loosen before spending.")
        return 1
    print("\nVERDICT: the rules admit competent play.")
    return 0


# Each case is a line lifted from the recorded Gemma rounds, turned into the
# state update that line would have produced. R1 subsumes the identity half of
# R2 and R3 -- an unlicensed binding change is caught whatever the outcome was --
# so R2 and R3 are tested on the part they alone cover: the objective.
DRIFT = [
    (
        "R1_BINDING_PERSISTS",
        '"Wait, i... the player is at (45-49, 34-38)" -- rebinding after a clean move',
        {"PLAYER_BINDING": "actually the orange block is the goal", "LAST_OUTCOME": "moved"},
    ),
    (
        "R2_NO_RESCAN_ON_MOVE",
        '"Let me look at the image again" -- new objective after a clean move',
        {"LOCAL_OBJECTIVE": "re-derive the whole scene", "LAST_OUTCOME": "moved"},
    ),
    (
        "R3_BLOCKED_IS_LOCAL",
        '"those are walls... let\'s re-examine the path" -- new objective after a wall',
        {"LOCAL_OBJECTIVE": "find a different goal entirely", "LAST_OUTCOME": "blocked"},
    ),
    (
        "R7_ONE_ACTION",
        "eight rounds of execute_python and no committed action",
        {"NEXT_STEP": None, "LAST_OUTCOME": "moved"},
    ),
    (
        "R8_CONTRADICTION_CONSUMED",
        "a contradiction opened, then left open as a standing licence to rethink",
        {"CONTRADICTION": "still unsure which object is the player", "LAST_OUTCOME": "moved"},
    ),
    (
        "R9_NO_STANDING_DOUBT",
        '"Wait, maybe it is this instead" -- third straight refusal to classify',
        {"LAST_OUTCOME": "uncertain"},
    ),
]


def replay_drift() -> int:
    """Does the controller catch the drift the transcripts actually show?"""

    _banner("replay: drift cases lifted from the gemma transcript")
    failures = 0
    for rule, description, violation in DRIFT:
        controller = Controller()
        base = {
            "PLAYER_BINDING": "player = two-colour block at 34,45",
            "LOCAL_OBJECTIVE": "reach shape changer",
            "ROUTE1": (1, 1, 3, 3),
            "CURRENT_APPROX_POSITION": "x34 y45",
            "LAST_ACTION": None,
            "LAST_OUTCOME": None,
            "NEXT_STEP": 1,
            "CONTRADICTION": "",
        }
        controller.propose(base, legal=LEGAL)
        # R9 needs the streak built up first.
        if rule == "R9_NO_STANDING_DOUBT":
            for _ in range(2):
                controller.propose(
                    {**base, "LAST_OUTCOME": "uncertain", "LAST_ACTION": 1}, legal=LEGAL
                )
        # R8 only fires on a contradiction left standing from the turn before.
        if rule == "R8_CONTRADICTION_CONSUMED":
            controller.propose(
                {**base, "LAST_OUTCOME": "moved", "LAST_ACTION": 1,
                 "CONTRADICTION": "the block I called the player did not move"},
                legal=LEGAL,
            )
        update = {**base, "LAST_ACTION": 1, **violation}
        try:
            controller.propose(update, legal=LEGAL)
        except Rejected as exc:
            mark = "ok " if exc.rule == rule else "MIS"
            if exc.rule != rule:
                failures += 1
            print(f"  {mark} {description}\n      caught by {exc.rule}")
            continue
        failures += 1
        print(f"  MISS {description}\n      expected {rule}, admitted")
    if failures:
        print(f"\nVERDICT: {failures} drift case(s) not caught as specified.")
        return 1
    print("\nVERDICT: every drift case is caught by its named rule.")
    return 0


def interactive() -> int:
    """Poke the state machine by hand."""

    controller = Controller()
    _banner("interactive controller")
    print(f"fields: {', '.join(FIELDS)}")
    print("enter 'FIELD=value' pairs separated by ';', blank line to quit.")
    print("ROUTE1 takes comma-separated ints. Example:")
    print("  PLAYER_BINDING=two-colour block; LOCAL_OBJECTIVE=changer; "
          "ROUTE1=1,1,3; NEXT_STEP=1; LAST_OUTCOME=moved\n")
    _show(controller)
    while True:
        try:
            line = input("\n> ").strip()
        except EOFError:
            break
        if not line:
            break
        update: dict[str, object] = {}
        for part in line.split(";"):
            if "=" not in part:
                continue
            key, _, value = part.partition("=")
            key, value = key.strip().upper(), value.strip()
            if key in {"NEXT_STEP", "LAST_ACTION"}:
                update[key] = int(value) if value.lstrip("-").isdigit() else None
            elif key == "ROUTE1":
                update[key] = tuple(int(v) for v in value.split(",") if v.strip())
            else:
                update[key] = value
        try:
            controller.propose(update, legal=LEGAL)
            _show(controller, f"accepted | {controller.metrics()}")
        except Rejected as exc:
            print(f"\033[31mREJECTED {exc.rule}\033[0m: {exc.detail}")
            _show(controller, "unchanged")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", choices=("luna", "drift", "both"))
    args = parser.parse_args()
    if args.replay == "luna":
        return replay_luna()
    if args.replay == "drift":
        return replay_drift()
    if args.replay == "both":
        return replay_luna() or replay_drift()
    return interactive()


if __name__ == "__main__":
    raise SystemExit(main())
