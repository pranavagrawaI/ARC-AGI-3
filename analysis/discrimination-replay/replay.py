#!/usr/bin/env python3
"""Would a discrimination trigger have opened on a run that already happened?

Offline, zero model spend. Replays a recorded NOOA run and asks, at every decision
point, whether a candidate trigger would have fired. This is a cheap filter for
KILLING bad triggers. It cannot show that a good one helps: it validates that a
trigger fires at the right time, never that firing changes the outcome.

    python3 replay.py --run <run-dir>                       # describe the substrate
    python3 replay.py --run <run-dir> --trigger stall_repetition
    python3 replay.py --two-sided --must-fire <fail-run> --must-not-flood <ok-run> \
        --before-step 188 --max-per-100 12

Two-sided is the point. A trigger tuned until it fires on one failure trace is a
detector for that trace. The must-not-flood side is what makes the must-fire side
mean something -- so a control that cannot exercise the trigger is reported
VACUOUS, never PASS. See README.md.

Substrate note: `events.jsonl` records `code_length`, not cell source. The belief
stream is only recoverable from `agent_logs/**/messages/`.
"""

from __future__ import annotations

# The runtime and scorer consume this same pure detector. Historical scorers stay frozen.
import sys
from pathlib import Path
_CONTRACT = Path(__file__).resolve().parents[1] / "local-probes/20260905T-stall-authority-q6/patched-source/after"

sys.path.insert(0, str(_CONTRACT))
from stall_contract import detect_stall

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

NOTES_TARGET = "hypotheses.md"

# ---------------------------------------------------------------- substrate


@dataclass(frozen=True)
class Hypothesis:
    hid: str
    status: str
    claim: str
    evidence: str

    @property
    def live(self) -> bool:
        """Anything the agent has not explicitly retired still governs behaviour."""
        return not any(
            dead in self.status.upper()
            for dead in ("REJECT", "DISPROV", "FALSE", "REFUTED", "CONTRADICT", "RULED OUT")
        )

    @property
    def confirmed(self) -> bool:
        return "CONFIRM" in self.status.upper()


@dataclass(frozen=True)
class Turn:
    """One agent decision: what it submitted and why."""

    step: int
    turn: int
    actions: list[str]
    rationale: str


@dataclass(frozen=True)
class State:
    """One harness-published observation."""

    step: int
    turn: int
    level: int
    levels_completed: int
    available_actions: list[str]
    state: str


@dataclass
class Run:
    run_dir: Path
    turns: list[Turn]
    states: list[State]
    beliefs: dict[int, list[Hypothesis]]  # RENDERED: step -> snapshot read back
    authored: dict[int, list[Hypothesis]]  # AUTHORED: step -> snapshot written
    writes: list[AuthoredWrite]
    stored: list[Hypothesis]  # STORED: final knowledge/hypotheses.md on disk
    rejections: list[dict]

    @property
    def name(self) -> str:
        return self.run_dir.name

    @property
    def max_step(self) -> int:
        return max([t.step for t in self.turns] + [s.step for s in self.states] + [0])

    def belief_at(self, step: int, source: str = "authored") -> list[Hypothesis]:
        """Forward-fill the chosen availability clock.

        `authored` is when the belief came into existence and entered the running
        transcript; `rendered` is when the agent last chose to read it back. The
        conflict question is about availability, so authored is the default.
        """
        table = self.authored if source == "authored" else self.beliefs
        seen = [s for s in sorted(table) if s <= step]
        return table[seen[-1]] if seen else []

    def state_at(self, step: int) -> State | None:
        prior = [s for s in self.states if s.step <= step]
        return prior[-1] if prior else None


_HYP_BLOCK = re.compile(r"stdout='''(# Hypotheses.*?)'''", re.DOTALL)
_HYP_ENTRY = re.compile(
    r"^\s*#{2,3}\s*(?P<hid>H\d+)\s*:\s*(?P<claim>.+?)\s*$", re.MULTILINE
)
_STATUS = re.compile(r"^\s*-?\s*Status:\s*(?P<status>.+?)\s*$", re.MULTILINE | re.IGNORECASE)
_EVIDENCE = re.compile(r"^\s*-?\s*Evidence:\s*(?P<ev>.+?)\s*$", re.MULTILINE | re.IGNORECASE)
_HYP_ROW = re.compile(
    r"^\s*\|\s*(?P<hid>H\d+)\s*\|\s*(?P<claim>[^|]*?)\s*\|"
    r"\s*(?P<status>[^|]*?)\s*\|\s*(?P<ev>[^|]*?)\s*\|",
    re.MULTILINE,
)
_MSG_STEP = re.compile(r"step_(\d+)_round_(\d+)_user\.md$")


def _parse_hypotheses(block: str) -> list[Hypothesis]:
    """Parse a hypotheses note in either shape the models actually use.

    The note format is model-chosen, not schema-enforced: q4 and q3c wrote
    `## H7: claim` headings with `- Status:` lines, q2d wrote a markdown table.
    Anything that has to read these notes must tolerate both -- which is itself an
    argument for giving the live discriminator a typed object instead.
    """
    entries: list[Hypothesis] = []
    matches = list(_HYP_ENTRY.finditer(block))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(block)
        body = block[m.end() : end]
        status = _STATUS.search(body)
        evidence = _EVIDENCE.search(body)
        entries.append(
            Hypothesis(
                hid=m.group("hid"),
                status=status.group("status") if status else "unstated",
                claim=m.group("claim").strip(),
                evidence=evidence.group("ev").strip() if evidence else "",
            )
        )
    for row in _HYP_ROW.finditer(block):
        entries.append(
            Hypothesis(
                hid=row.group("hid"),
                status=row.group("status").strip() or "unstated",
                claim=row.group("claim").strip(),
                evidence=row.group("ev").strip(),
            )
        )
    return entries


@dataclass(frozen=True)
class AuthoredWrite:
    """One `append_notes` / `write_notes` call, recovered from the cell source.

    This is the earliest moment a belief exists. Readback is a later, optional
    event, so an availability clock built on readback understates by however long
    the agent went without looking.
    """

    step: int
    round: int
    fn: str
    target: str
    payload: str


_TOOL_ARGS = re.compile(r"^\[tool_call\]\s+name=\S+\s+args=(\{.*\})\s*$", re.MULTILINE)
_MSG_ASSISTANT = re.compile(r"step_(\d+)_round_(\d+)_assistant\.md$")
_NOTE_FNS = frozenset({"append_notes", "write_notes"})


def _note_writes(code: str) -> list[tuple[str, str, str]]:
    """(fn, target, payload) for every note write in one cell, via AST not regex.

    An AST walk gets the payload regardless of quoting style and ignores writes
    that only appear inside a string the cell happens to print.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    found: list[tuple[str, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in _NOTE_FNS or len(node.args) < 2:
            continue
        target, payload = node.args[0], node.args[1]
        if not (isinstance(target, ast.Constant) and isinstance(target.value, str)):
            continue
        if not (isinstance(payload, ast.Constant) and isinstance(payload.value, str)):
            continue
        found.append((node.func.attr, target.value, payload.value))
    return found


def _load_authored(messages: Path) -> list[AuthoredWrite]:
    writes: list[AuthoredWrite] = []
    for path in sorted(messages.glob("*_assistant.md")):
        m = _MSG_ASSISTANT.search(path.name)
        if m is None:
            continue
        step, rnd = int(m.group(1)), int(m.group(2))
        for raw in _TOOL_ARGS.findall(path.read_text()):
            try:
                code = json.loads(raw).get("code", "")
            except json.JSONDecodeError:
                continue
            for fn, target, payload in _note_writes(code):
                writes.append(
                    AuthoredWrite(step=step, round=rnd, fn=fn, target=target, payload=payload)
                )
    return sorted(writes, key=lambda w: (w.step, w.round))


def _authored_timeline(
    writes: list[AuthoredWrite], target: str
) -> dict[int, list[Hypothesis]]:
    """Replay note writes in order into a step -> cumulative snapshot map.

    `write_notes` replaces the file, `append_notes` extends it; later entries for
    the same id supersede earlier ones, which is how a status reaches CONFIRMED.
    """
    timeline: dict[int, list[Hypothesis]] = {}
    current: dict[str, Hypothesis] = {}
    for write in writes:
        if write.target != target:
            continue
        if write.fn == "write_notes":
            current = {}
        for hypothesis in _parse_hypotheses(write.payload):
            current[hypothesis.hid] = hypothesis
        timeline[write.step] = list(current.values())
    return timeline


def _messages_dir(run_dir: Path) -> Path | None:
    return next(iter(sorted(run_dir.glob("agent_logs/*/*/messages"))), None)


def _events_path(run_dir: Path) -> Path | None:
    return next(iter(sorted(run_dir.glob("agent_logs/*/*/events.jsonl"))), None)


def _load_jsonl(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_run(run_dir: Path) -> Run:
    """Assemble the per-step record from the three places it actually lives."""
    events = _load_jsonl(_events_path(run_dir))
    turns = [
        Turn(
            step=int(e["step"]),
            turn=int(e.get("turn", -1)),
            actions=list(e.get("actions") or []),
            rationale=str(e.get("rationale") or ""),
        )
        for e in events
        if e.get("event") == "agent_turn"
    ]

    states = [
        State(
            step=int(s.get("step", 0)),
            turn=int(s.get("turn", 0)),
            level=int(s.get("level", 0)),
            levels_completed=int(s.get("levels_completed", 0)),
            available_actions=list(s.get("available_actions") or []),
            state=str(s.get("state", "")),
        )
        for s in _load_jsonl(run_dir / "ipc" / "states.jsonl")
    ]

    # The belief stream reaches the model only when a cell reads the file back, so
    # a snapshot exists at a step only if the agent chose to look. That absence is
    # itself a measurement -- do not synthesise from the on-disk final file.
    beliefs: dict[int, list[Hypothesis]] = {}
    messages = _messages_dir(run_dir)
    if messages is not None:
        for path in sorted(messages.glob("*_user.md")):
            m = _MSG_STEP.search(path.name)
            if m is None:
                continue
            blocks = _HYP_BLOCK.findall(path.read_text())
            if not blocks:
                continue
            step = int(m.group(1))
            parsed = _parse_hypotheses(blocks[-1])
            if parsed and len(parsed) >= len(beliefs.get(step, [])):
                beliefs[step] = parsed

    writes = _load_authored(messages) if messages is not None else []
    stored_path = next(
        iter(sorted(run_dir.glob(f"**/knowledge/{NOTES_TARGET}"))), None
    )
    stored = _parse_hypotheses(stored_path.read_text()) if stored_path else []

    return Run(
        run_dir=run_dir,
        turns=sorted(turns, key=lambda t: t.step),
        states=sorted(states, key=lambda s: s.step),
        beliefs=beliefs,
        authored=_authored_timeline(writes, NOTES_TARGET),
        writes=writes,
        stored=stored,
        rejections=_load_jsonl(run_dir / "ipc" / "rejections.jsonl"),
    )


# ----------------------------------------------------------------- triggers


@dataclass(frozen=True)
class Opening:
    """One firing, with everything a human needs to audit it by hand."""

    step: int
    turn: int
    reason: str
    hypotheses: list[str] = field(default_factory=list)
    probe_hint: str = ""

    def render(self) -> str:
        who = f" [{', '.join(self.hypotheses)}]" if self.hypotheses else ""
        hint = f"\n      probe: {self.probe_hint}" if self.probe_hint else ""
        return f"  step {self.step:>4} turn {self.turn:>3}{who}\n      {self.reason}{hint}"


_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def trigger_stall_repetition(
    run: Run,
    *,
    window: int = 6,
    min_actions: int = 25,
    similarity: float = 0.55,
    requires_beliefs: bool = False,
) -> list[Opening]:
    """Semantics-free. Progress stalled, intent repeating, options untried.

    Deliberately reads no hypotheses, so it is the only trigger a control run
    without a belief stream can legitimately exercise.
    """
    openings: list[Opening] = []
    for i, turn in enumerate(run.turns):
        if i < window:
            continue
        state = run.state_at(turn.step)
        if state is None or turn.step < min_actions:
            continue

        recent = run.turns[i - window : i]
        earlier_state = run.state_at(recent[0].step)
        if earlier_state is None:
            continue
        evidence = detect_stall(
            [{"actions": p.actions, "rationale": p.rationale,
              "levels_completed": (run.state_at(p.step) or earlier_state).levels_completed}
             for p in recent],
            {"step": turn.step, "turn": turn.turn, "levels_completed": state.levels_completed,
             "available_actions": state.available_actions},
            turn.rationale, window=window, min_actions=min_actions, similarity=similarity,
        )
        if evidence is None:
            continue
        best = evidence["similarity"]
        untried = evidence["untried_actions"]

        openings.append(
            Opening(
                step=turn.step,
                turn=turn.turn,
                reason=(
                    f"no level in {window} turns, {turn.step} actions spent; "
                    f"rationale repeats prior turn at {best:.2f} similarity"
                ),
                probe_hint=f"untried from here: {', '.join(untried)}",
            )
        )
    return openings


_NEGATION = re.compile(
    r"\b(?:not|never|no longer|isn't|aren't|is not|are not)\b", re.IGNORECASE
)
_SUBJECT_SPLIT = re.compile(r"\s*(?:=|:|\bare\b|\bis\b)\s*", re.IGNORECASE)
_STOP = frozenset(
    ["a", "an", "the", "of", "at", "in", "on", "to", "and", "or", "for", "with", "is", "are", "was", "were", "be", "not", "no", "this", "that", "these", "those", "it", "its", "from", "by", "as", "into", "different"]
)


def _content(text: str) -> set[str]:
    return {t for t in _tokens(text) if t not in _STOP and len(t) > 2}


def trigger_hypothesis_conflict(
    run: Run,
    *,
    window: int = 4,
    requires_beliefs: bool = True,
) -> list[Opening]:
    """A CONFIRMED hypothesis negates a term a live hypothesis still asserts, and
    the agent's own recent rationale is still citing that term.

    Every firing names both hypotheses and the overlapping token, so a reviewer can
    reject it by hand. That auditability is the point: this rule is a heuristic
    tuned on one trace and must be treated as such.
    """
    openings: list[Opening] = []
    fired: set[tuple[str, str]] = set()

    for i, turn in enumerate(run.turns):
        snapshot = run.belief_at(turn.step)
        if not snapshot:
            continue
        recent = run.turns[max(0, i - window) : i + 1]
        rationale_tokens = _content(" ".join(p.rationale for p in recent))
        if not rationale_tokens:
            continue

        for confirmed in (h for h in snapshot if h.confirmed and _NEGATION.search(h.claim)):
            negated = _content(_NEGATION.split(confirmed.claim)[-1])
            if not negated:
                continue
            for other in snapshot:
                if other.hid == confirmed.hid or not other.live or other.confirmed:
                    continue
                shared = negated & _content(other.claim) & rationale_tokens
                if not shared:
                    continue
                key = (confirmed.hid, other.hid)
                if key in fired:
                    continue
                fired.add(key)
                token = ", ".join(sorted(shared))
                openings.append(
                    Opening(
                        step=turn.step,
                        turn=turn.turn,
                        reason=(
                            f"{confirmed.hid} ({confirmed.status}) denies "
                            f"'{token}'; {other.hid} ({other.status}) still asserts it "
                            f"and the last {len(recent)} rationales still cite it"
                        ),
                        hypotheses=[confirmed.hid, other.hid],
                        probe_hint=(
                            f"an action whose outcome separates {confirmed.hid} "
                            f"from {other.hid} on '{token}'"
                        ),
                    )
                )
    return openings


def trigger_persistent_conflict(
    run: Run,
    *,
    window: int = 4,
    persistence: int = 6,
    requires_beliefs: bool = True,
) -> list[Opening]:
    """A conflict that is still unresolved `persistence` turns after it appeared.

    `hypothesis_conflict` fires more often on the successful q2d than on the
    failing q4, so the mere existence of a contradiction is not pathological --
    an active investigator generates and retires them constantly. What separated
    the runs was whether the contradiction ever took over. This adds that term:
    the same pair must still be live, still contradictory, and still cited that
    many turns later.
    """
    raw = trigger_hypothesis_conflict(run, window=window)
    turns_by_step = {t.step: i for i, t in enumerate(run.turns)}
    openings: list[Opening] = []

    for opening in raw:
        start = turns_by_step.get(opening.step)
        if start is None or start + persistence >= len(run.turns):
            continue
        confirmed_id, other_id = opening.hypotheses
        later = run.turns[start + persistence]
        snapshot = {h.hid: h for h in run.belief_at(later.step)}
        confirmed, other = snapshot.get(confirmed_id), snapshot.get(other_id)
        if confirmed is None or other is None or not other.live or other.confirmed:
            continue  # the agent retired or promoted it: authority did transfer

        denied = _content(_NEGATION.split(confirmed.claim)[-1])
        window_turns = run.turns[start : start + persistence + 1]
        cited = _content(" ".join(t.rationale for t in window_turns))
        shared = denied & _content(other.claim) & cited
        if not shared:
            continue  # the agent stopped acting on it, even without editing the note

        openings.append(
            Opening(
                step=later.step,
                turn=later.turn,
                reason=(
                    f"{confirmed_id} vs {other_id} on '{', '.join(sorted(shared))}' "
                    f"still unresolved {persistence} turns after step {opening.step}; "
                    f"{other_id} still {other.status} and still driving rationales"
                ),
                hypotheses=list(opening.hypotheses),
                probe_hint=opening.probe_hint,
            )
        )
    return openings


TRIGGERS = {
    "stall_repetition": trigger_stall_repetition,
    "hypothesis_conflict": trigger_hypothesis_conflict,
    "persistent_conflict": trigger_persistent_conflict,
}


def requires_beliefs(name: str) -> bool:
    return bool(TRIGGERS[name].__kwdefaults__.get("requires_beliefs"))


# ------------------------------------------------------------------ reports


def integrity(run: Run) -> str:
    """Diff AUTHORED -> STORED -> RENDERED for every hypothesis.

    The invariant: any mutation of agent-authored semantic content between write
    and render is an integrity failure unless the transformed span is marked,
    attributed and reconstructible. Redaction may be necessary; invisible
    semantic rewriting is not.
    """
    authored = {h.hid: h for step in sorted(run.authored) for h in run.authored[step]}
    stored = {h.hid: h for h in run.stored}
    rendered_steps = sorted(run.beliefs)
    rendered = (
        {h.hid: h for h in run.beliefs[rendered_steps[-1]]} if rendered_steps else {}
    )

    lines = [
        f"integrity  {run.name}",
        f"  AUTHORED {len(authored)} hypotheses from {len(run.writes)} note writes",
        f"  STORED   {len(stored)} in knowledge/{NOTES_TARGET}",
        f"  RENDERED {len(rendered)} in the last readback",
        "",
    ]
    faults = 0
    for hid in sorted(authored, key=lambda h: int(h[1:])):
        for label, table in (("stored", stored), ("rendered", rendered)):
            other = table.get(hid)
            if other is None or other.claim == authored[hid].claim:
                continue
            faults += 1
            lines += [
                f"  MUTATED {hid} authored -> {label}",
                f"    authored: {authored[hid].claim}",
                f"    {label:<8}: {other.claim}",
            ]
    missing = [h for h in authored if h not in stored and stored]
    if missing:
        lines.append(f"  DROPPED before storage: {', '.join(sorted(missing))}")
    lines.append("")
    lines.append(
        f"  {faults} semantic mutation(s) of actor-authored belief"
        + ("  -> INTEGRITY FAIL" if faults else "  -> clean")
    )
    return "\n".join(lines)


def describe(run: Run) -> str:
    steps = sorted(run.beliefs)
    span = f"{steps[0]}..{steps[-1]}" if steps else "none"
    total = sum(len(v) for v in run.beliefs.values())
    levels = max((s.levels_completed for s in run.states), default=0)
    lines = [
        f"run              {run.name}",
        f"agent turns      {len(run.turns)}",
        f"published states {len(run.states)}",
        f"env steps (max)  {run.max_step}",
        f"levels completed {levels}",
        f"host rejections  {len(run.rejections)}",
        (
            f"belief snapshots {len(run.beliefs)} (steps {span}), "
            f"{total} parsed hypothesis entries"
        ),
    ]
    astep = sorted(run.authored)
    aspan = f"{astep[0]}..{astep[-1]}" if astep else "none"
    lines.append(
        f"authored belief  {len(run.writes)} note writes, snapshots at steps {aspan}"
    )
    if run.authored:
        last = run.authored[astep[-1]]
        confirmed = [h.hid for h in last if h.confirmed]
        lines.append(
            f"final authored   {len(last)} hypotheses, confirmed: "
            f"{', '.join(confirmed) or 'none'}"
        )
        gap = (steps[0] - astep[0]) if steps else None
        if gap:
            lines.append(
                f"readback lag     {gap} steps between first authored belief and "
                f"first readback"
            )
    if not run.beliefs:
        lines.append(
            "rendered belief  NONE -- this run never read its hypotheses back into "
            "context; rendered-source triggers cannot be exercised on it"
        )
    return "\n".join(lines)


def _verdict(passed: bool | None) -> str:
    return {True: "PASS", False: "FAIL", None: "VACUOUS"}[passed]


def two_sided(
    must_fire: Run,
    must_not_flood: Run,
    trigger: str,
    *,
    before_step: int,
    max_per_100: float,
) -> tuple[str, str]:
    """Sensitivity on the failure trace, specificity on the success trace.

    A control that cannot exercise the trigger yields VACUOUS, because a trigger
    that never fires for want of input is indistinguishable from a specific one.
    """
    fn = TRIGGERS[trigger]
    fire = fn(must_fire)
    flood = fn(must_not_flood)

    early = [o for o in fire if o.step < before_step]
    sensitivity = bool(early)

    control_blind = requires_beliefs(trigger) and not must_not_flood.authored
    span = max(must_not_flood.max_step, 1)
    rate = len(flood) * 100.0 / span
    specificity = None if control_blind else rate <= max_per_100

    fire_span = max(must_fire.max_step, 1)
    fire_rate = len(fire) * 100.0 / fire_span
    out = [
        f"trigger: {trigger}",
        "",
        f"MUST FIRE   {must_fire.name}",
        (
            f"  {len(fire)} openings, {len(early)} before step {before_step}"
            f"  -> {_verdict(sensitivity)}"
        ),
    ]
    out += [o.render() for o in early[:5]]
    if len(early) > 5:
        out.append(f"  ... {len(early) - 5} more")

    out += [
        "",
        f"MUST NOT FLOOD  {must_not_flood.name}",
        (
            f"  {len(flood)} openings over {span} steps = {rate:.1f} per 100 "
            f"(limit {max_per_100})  -> {_verdict(specificity)}"
        ),
    ]
    if control_blind:
        out.append(
            "  control has no belief stream: this trigger cannot fire here for lack "
            "of input, not for specificity. The result carries no information."
        )
    out += [o.render() for o in flood[:3]]

    # Separation is the measurement; the per-100 limit is a threshold someone chose.
    # A trigger that fires no more densely on the failure than on the success is not
    # discriminating, whatever an arbitrary cap says about it.
    separation = fire_rate / rate if rate else float("inf")
    discriminating = separation > 1.0
    out += [
        "",
        "SEPARATION",
        f"  must-fire      {fire_rate:5.1f} openings per 100 steps",
        f"  must-not-flood {rate:5.1f} openings per 100 steps",
        f"  ratio          {separation:5.2f}x"
        + ("" if discriminating else "   <- fires no harder on the failure"),
    ]

    if specificity is None:
        overall = "VACUOUS"
    elif sensitivity and specificity and discriminating:
        overall = "PASS"
    else:
        overall = "FAIL"
    out += ["", f"VERDICT: {overall}"]
    return "\n".join(out), overall


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, help="describe or evaluate a single run")
    ap.add_argument("--trigger", choices=sorted(TRIGGERS))
    ap.add_argument("--two-sided", action="store_true")
    ap.add_argument("--must-fire", type=Path)
    ap.add_argument("--must-not-flood", type=Path)
    ap.add_argument("--before-step", type=int, default=188)
    ap.add_argument("--max-per-100", type=float, default=12.0)
    ap.add_argument("--json", type=Path, help="write the openings as JSON")
    ap.add_argument(
        "--integrity", action="store_true", help="diff AUTHORED -> STORED -> RENDERED"
    )
    args = ap.parse_args()

    if args.two_sided:
        if not (args.must_fire and args.must_not_flood and args.trigger):
            ap.error("--two-sided needs --must-fire, --must-not-flood and --trigger")
        report, overall = two_sided(
            load_run(args.must_fire),
            load_run(args.must_not_flood),
            args.trigger,
            before_step=args.before_step,
            max_per_100=args.max_per_100,
        )
        print(report)
        return 0 if overall == "PASS" else 1

    if not args.run:
        ap.error("pass --run or --two-sided")
    run = load_run(args.run)
    if args.integrity:
        print(integrity(run))
        return 0
    if not args.trigger:
        print(describe(run))
        return 0

    openings = TRIGGERS[args.trigger](run)
    print(f"{args.trigger}: {len(openings)} openings on {run.name}")
    for opening in openings:
        print(opening.render())
    if args.json:
        args.json.write_text(
            json.dumps([o.__dict__ for o in openings], indent=2) + "\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
