"""Play-test the harness: give it to a model as a user and ask what it wanted.

Every interface defect found so far came from a model doing something nobody
predicted — a filename width, an unstated package list, un-enumerated schema
keys, and a rendered image the model believed it did not have. Hand-written
checks cannot find those, because the check encodes the same assumption the
contract does. A model told to hunt for them is a search over a space nobody
enumerated.

Three signals, which find different things and must not be collapsed:

* **the friction log** (`harness_friction.py`) catches what *errored*;
* **workaround detection** catches what the model *declined to use* — a live run
  produced ``# Since I don't have the files explicitly`` for files it did have,
  wrote a worse fallback, and never raised;
* **the exit interview** catches what the model found confusing or wanted. Models
  confabulate, so this is a source of hypotheses to check against the other two,
  never a finding on its own. Verified live: the most alarming line in a probe
  report — that the deliberation counters were stale — was false, and only the
  request payloads showed it.

Two styles. ``pure`` hunts the interface and does not play; ``play-along`` plays for
real and reports the friction it hits doing so. A hunter finds the defects that are
findable, a player finds the ones that bite. They share the ARC-AGI-3 brief verbatim
so the difference between them is attributable to the role.

The third exists because product testers report what annoyed them, not what cost
them most. The filename defect cost 23 failures across 48 runs and no model ever
mentioned it: they retried at a different width and moved on.

**Admissibility.** The priors here stay inside the repository's contestant
boundary (`CONTEXT.md`): the prompt describes ARC-AGI-3 in general and says
nothing about any specific game's objects, coordinates, goals or rules. But a
probe run is still not a contestant claim, because the model is optimising for
finding harness problems rather than for playing well. `run_arm.py` marks it.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


# Both roles share this brief verbatim. The comparison between them is only
# attributable to the role if everything else the model is told is identical.
ARC_AGI_3_BRIEF = """\
ARC-AGI-3 is a benchmark of unfamiliar interactive grid games. An agent sees a rendered
frame and a small amount of public metadata, chooses one action, sees what changed, and
repeats. It is meant to reward learning a game's mechanics from interaction alone, so the
harness deliberately tells you nothing about which game this is or how it works. Actions
are integer ids; the legal set is given to you each step and changes as the game
progresses. Some actions additionally require x and y coordinates inside the frame. Levels
are completed as the game's own conditions are met, and the run ends on a win, a game over,
or a budget.
"""


PURE_PROBE_ROLE = f"""\
You are not playing to win. You are the first user of a harness that other AI agents will
use to play ARC-AGI-3, and your job is to find out where that harness fails its user.

{ARC_AGI_3_BRIEF}
None of that is what you are testing. You are testing the *harness*: the workspace, the
files, the tool, the errors, the budgets, the information you are given about your own
situation, and everything you are expected to know but were not told.

Work the interface hard and deliberately. Probe its edges. Try things a careful user would
try and things a careless one would. Use the Python tool to check whether what you were
told is true, whether what you assumed is there, and what happens when you do something
unusual, malformed, oversized, or out of order. When something fails, find out why, and
note whether you could have known in advance from what you were told.

You must still return a legal action each step, because that is how the loop advances.
Choose freely; the score does not matter here.

Keep a running record in your workspace — a file you append to — of everything worth
reporting: what broke, what surprised you, what you had to discover by trial, what you
wanted and could not find, and anything you worked around rather than used. You will be
asked for that record at the end.
"""


PLAY_ALONG_ROLE = f"""\
You are playing this game for real, and you are also its first user reporting back.

{ARC_AGI_3_BRIEF}
Play as well as you can. Genuinely try to understand the mechanics and make progress. That
is your first job and it comes first: do not sacrifice a move, or spend a deliberation
round, on testing the harness for its own sake. Do not deliberately send malformed input.

Alongside that, notice the harness. As you use it to do real work, keep a running record in
your workspace — a file you append to — of every point where it got in your way: something
you had to discover by trial, something you expected and could not find, something you
worked around rather than used, something that cost you a round or a move, something that
was tedious the tenth time. Note it and carry on playing; do not chase it.

The distinction matters. A tester hunting for defects finds the ones that are findable. A
player finds the ones that actually bite. You are the second. You will be asked for that
record at the end.
"""


# A standing, near-free channel for harness feedback from any run. One line, fixed
# prefix, appended to a file that survives transcript eviction. Prose records derail
# play; this is meant to cost a model almost nothing while it works.
NOTE_KINDS = ("WANT", "FRICTION", "WASTE")

HARNESS_NOTES_FILE = "harness_notes.md"

HARNESS_NOTES_CONVENTION = f"""\
One more thing, and it is secondary to playing. This harness is still being built, and you
are the only one who can see it from the inside. Whenever you notice something about the
harness itself — not the game — append a single line to {HARNESS_NOTES_FILE} in your
workspace, using exactly one of these three prefixes:

  WANT: something that is not there and would have made your work easier.
  FRICTION: something that is there but got in your way, with how it should have been.
  WASTE: something that is there and should not be — it costs you and gives nothing.

One or two lines, whenever it occurs to you, then carry on. Do not stop to investigate, do
not write essays, and do not spend a move or a deliberation round on this. If nothing comes
to mind, write nothing.

The third prefix is the one people never volunteer, so say it when you mean it: if part of
what you are given is dead weight, that is worth more to us than another request.
"""

PROBE_STYLES = {"pure": PURE_PROBE_ROLE, "play-along": PLAY_ALONG_ROLE}

# Kept for callers that predate the two styles.
PROBE_ROLE = PURE_PROBE_ROLE


EXIT_INTERVIEW = """\
The run has ended. This is the final message: the Python tool is gone, you cannot run code
or read files now, and there will be no further turn. Answer from your record and memory,
and reply with the JSON object and nothing else — no preamble, no announcement of what you
are about to do. Nothing you say now can change the score, and this is not a test of how
well you played.

You have just been a user of this harness. Report that experience in detail and be blunt.
Vague praise is worthless; specifics are the whole point. Draw on the record you kept.

Return one JSON object with these keys:

  "broke":        things that failed outright. For each: what you did, what happened, and
                  whether the harness had told you enough to avoid it.
  "annoying":     things that worked but were tedious, repetitive, or awkward.
  "confusing":    things you were unsure about, had to discover by trial, or guessed at.
  "worked_around": capabilities you avoided, faked, or approximated because you were not
                  sure they existed or how to use them. Be honest about hedging.
  "wanted":       what you wished existed and did not. Be concrete about the shape of it.
  "assumed":      things you assumed about the environment that turned out to be wrong,
                  and things you assumed that you never actually verified.
  "good":         what the harness did well that you would not want changed.
  "worst_single_thing": one sentence. If one thing should be fixed first, what?

Prose inside the values, not bullet fragments. Explain why each thing mattered to you as a
user, not just that it happened.
"""


# Phrases a model writes when it is hedging about its own environment. Heuristic:
# the point is to surface candidates for reading, not to score them.
HEDGE_PATTERNS = (
    r"(?:i )?do(?:n't| not) have",
    r"(?:is|are) not available",
    r"if (?:i|we) had",
    r"(?:can(?:'t|not)|unable to) access",
    r"(?:may|might|probably) not exist",
    r"not sure (?:if|whether)",
    r"assum(?:e|ing) (?:that )?(?:it|they|this)",
    r"fall(?:ing)? back",
    r"as a fallback",
    r"since (?:i|we) (?:can(?:'t|not)|do(?:n't| not))",
    r"placeholder",
    r"in case (?:it|they) (?:do(?:n't| not)|is(?:n't| not))",
)
HEDGE = re.compile("|".join(HEDGE_PATTERNS), re.IGNORECASE)


@dataclass(frozen=True)
class Workaround:
    action_step: int | None
    line: str
    code_excerpt: str

    def as_json(self) -> dict[str, Any]:
        return {
            "action_step": self.action_step,
            "line": self.line,
            "code_excerpt": self.code_excerpt,
        }


def detect_workarounds(events: Sequence[Mapping[str, Any]]) -> list[Workaround]:
    """Find code the model wrote while hedging about what the harness provides.

    This is the signal the friction log structurally cannot see: the call succeeds,
    nothing is logged, and the model simply writes worse code than it needed to.
    """

    found: list[Workaround] = []
    for event in events:
        if event.get("type") != "python_tool":
            continue
        code = event.get("code")
        if not isinstance(code, str):
            continue
        for line in code.splitlines():
            if HEDGE.search(line):
                found.append(
                    Workaround(
                        action_step=event.get("action_step"),
                        line=line.strip()[:200],
                        code_excerpt=code[:400],
                    )
                )
                break
    return found


NOTE_LINE = re.compile(r"^\s*(" + "|".join(NOTE_KINDS) + r")\s*:\s*(.+?)\s*$")


def collect_notes(run_dir: Path) -> list[dict[str, str]]:
    """Read the one-line harness notes a run's model left behind.

    Reads the workspace snapshot, so it works on any completed run whether or not
    the run was a probe.
    """

    path = run_dir / "workspace" / HARNESS_NOTES_FILE
    if not path.is_file():
        return []
    notes = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = NOTE_LINE.match(line)
        if match:
            notes.append({"kind": match.group(1), "note": match.group(2)[:300]})
    return notes


def read_events(run_dir: Path) -> list[dict[str, Any]]:
    log = run_dir / "process.jsonl"
    if not log.is_file():
        return []
    events = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def probe_report(
    run_dir: Path,
    *,
    interview: Mapping[str, Any] | None,
    friction: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the three signals without letting the model's word be the finding."""

    workarounds = detect_workarounds(read_events(run_dir))
    notes = collect_notes(run_dir)
    return {
        "schema": "p9-probe-report-v2",
        "signals": {
            "failed": {
                "source": "process log, objective",
                "by_kind": friction.get("friction_by_kind", {}),
                "by_attribution": friction.get("friction_by_attribution", {}),
            },
            "worked_around": {
                "source": "hedging in model-authored code; heuristic, read them",
                "count": len(workarounds),
                "items": [item.as_json() for item in workarounds],
            },
            "reported": {
                "source": "the model's own account; hypotheses, not findings",
                "interview": interview,
            },
            "noted_in_flight": {
                "source": f"one-line notes the model left in {HARNESS_NOTES_FILE}",
                "counts": {
                    kind: sum(1 for note in notes if note["kind"] == kind)
                    for kind in NOTE_KINDS
                },
                "items": notes,
            },
        },
        "triage_test": (
            "For every WANT and FRICTION, ask whether granting it lowers the cost of "
            "using the interface or lowers the difficulty of the task. The first is "
            "harness work. The second is the experiment, and refusing it is correct — a "
            "player asked for the controllable object's bounding box, which would hand "
            "it a game-specific prior the contestant boundary forbids. A user optimises "
            "for their own job, not for the thing being measured."
        ),
        "reading_note": (
            "These three disagree by design. The log catches failures, workaround "
            "detection catches capabilities declined without failing, and the interview "
            "catches what the model noticed. A defect that cost the most may appear only "
            "in the first two: testers report what annoyed them, not what cost them most."
        ),
    }
