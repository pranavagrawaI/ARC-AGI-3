"""0012 — Depth-gated clause search over the closed alphabet.

Run from the repo root:
    uv run python teach/lessons/0012-depth-gated-clause-search.py

Lessons 0010/0011 turn frames into confidence-gated event facts. This lesson
learns ONE deterministic rule from those facts without naming game roles.

GIVEN: a fixed observational alphabet, conjunction, a bounded maximum depth,
       and an MDL search prior.
LEARNED: which observed literals jointly predict an effect in this game.

The search is exhaustive because the grammar is intentionally tiny. It deepens
only when every shallower clause makes mistakes; it stops at the first depth
with an exact explanation. If no clause fits by max_depth, it reports
GRAMMAR_GAP instead of returning the least-bad story.

UNCERTAIN and LOST identity events are excluded: a guessed correspondence or an
expired memory hypothesis must not create or kill a world law.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

UNTRUSTED_STATUSES = frozenset({"UNCERTAIN", "LOST"})


@dataclass(frozen=True)
class Event:
    facts: frozenset[str]
    effect: bool
    status: str = "OBSERVED"


@dataclass(frozen=True)
class Clause:
    literals: tuple[str, ...]

    def predicts(self, event: Event) -> bool:
        return set(self.literals) <= event.facts

    def errors(self, events: list[Event]) -> int:
        return sum(self.predicts(event) != event.effect for event in events)

    def __str__(self) -> str:
        body = "TRUE" if not self.literals else " AND ".join(self.literals)
        return f"IF {body} THEN effect"


@dataclass(frozen=True)
class DepthReport:
    depth: int
    candidates: int
    best_clause: Clause
    errors: int
    mdl_bits: int


@dataclass(frozen=True)
class SearchResult:
    status: str
    clause: Clause | None
    trusted_events: int
    filtered_events: int
    explored_candidates: int
    depth_reports: tuple[DepthReport, ...]


def trusted(events: list[Event]) -> list[Event]:
    return [event for event in events if event.status not in UNTRUSTED_STATUSES]


def mdl_bits(clause: Clause, events: list[Event], vocabulary_size: int) -> int:
    """Small two-part-code proxy: encode the rule, then its residual mistakes.

    This is pedagogical MDL, not normalized maximum likelihood. A literal costs
    enough bits to select one vocabulary item; each mistake costs enough bits to
    identify one event. Exact short rules therefore beat exact long rules, while
    a slightly shorter rule may still lose if it leaves unexplained data.
    """
    literal_bits = max(1, math.ceil(math.log2(max(2, vocabulary_size))))
    error_bits = max(1, math.ceil(math.log2(len(events) + 1)))
    return 1 + len(clause.literals) * literal_bits + clause.errors(events) * error_bits


def learn_clause(events: list[Event], max_depth: int = 3) -> SearchResult:
    clean = trusted(events)
    if not clean:
        raise ValueError("no trusted events")
    if not any(event.effect for event in clean):
        raise ValueError("need at least one positive effect event")
    if not any(not event.effect for event in clean):
        raise ValueError("need at least one negative effect event")
    if max_depth < 0:
        raise ValueError("max_depth must be >= 0")

    vocabulary = sorted(set().union(*(event.facts for event in clean)))
    reports: list[DepthReport] = []
    explored = 0

    for depth in range(0, min(max_depth, len(vocabulary)) + 1):
        candidates = [Clause(tuple(parts)) for parts in combinations(vocabulary, depth)]
        explored += len(candidates)
        ranked = sorted(
            candidates,
            key=lambda clause: (
                clause.errors(clean),
                mdl_bits(clause, clean, len(vocabulary)),
                clause.literals,
            ),
        )
        best = ranked[0]
        report = DepthReport(
            depth=depth,
            candidates=len(candidates),
            best_clause=best,
            errors=best.errors(clean),
            mdl_bits=mdl_bits(best, clean, len(vocabulary)),
        )
        reports.append(report)

        # Depth gate: do not enumerate more expressive clauses after a perfect
        # explanation exists. Within this depth, MDL/lexical order breaks ties.
        exact = [clause for clause in candidates if clause.errors(clean) == 0]
        if exact:
            winner = min(
                exact,
                key=lambda clause: (
                    mdl_bits(clause, clean, len(vocabulary)),
                    clause.literals,
                ),
            )
            return SearchResult(
                status="LEARNED",
                clause=winner,
                trusted_events=len(clean),
                filtered_events=len(events) - len(clean),
                explored_candidates=explored,
                depth_reports=tuple(reports),
            )

    return SearchResult(
        status="GRAMMAR_GAP",
        clause=None,
        trusted_events=len(clean),
        filtered_events=len(events) - len(clean),
        explored_candidates=explored,
        depth_reports=tuple(reports),
    )


def _training_trace() -> list[Event]:
    """Generic observations; names describe slots, never ls20 object roles."""
    return [
        Event(frozenset({"contact", "target_shape=cross", "actor_color=blue", "action=1", "target_visible"}), True),
        Event(frozenset({"contact", "target_shape=cross", "actor_color=green", "action=2", "target_visible"}), True),
        Event(frozenset({"contact", "target_shape=square", "actor_color=blue", "action=1", "target_visible"}), False),
        Event(frozenset({"near", "target_shape=cross", "actor_color=blue", "action=1", "target_visible"}), False),
        Event(frozenset({"aligned", "target_shape=square", "actor_color=green", "action=2", "target_visible"}), False),
        # Contradictory guessed identity: confidence gating must remove it.
        Event(frozenset({"contact", "target_shape=cross", "actor_color=blue", "action=1", "target_visible"}), False, "UNCERTAIN"),
        Event(frozenset({"contact", "target_shape=cross", "actor_color=blue", "action=1", "target_visible"}), False, "LOST"),
    ]


def _check(name: str, ok: bool, detail: str) -> bool:
    print(f"  [{'✓' if ok else '✗'}] {name}: {detail}")
    return ok


def _run_checks() -> bool:
    trace = _training_trace()
    result = learn_clause(trace, max_depth=3)
    checks = []

    checks.append(_check(
        "1 confidence gate",
        result.trusted_events == 5 and result.filtered_events == 2,
        f"trusted={result.trusted_events}, filtered={result.filtered_events} (exp 5,2)",
    ))
    checks.append(_check(
        "2 depth one is insufficient",
        result.depth_reports[1].errors > 0,
        f"best depth-1 errors={result.depth_reports[1].errors} (exp >0)",
    ))
    expected = ("contact", "target_shape=cross")
    checks.append(_check(
        "3 learns the minimal conjunction",
        result.status == "LEARNED" and result.clause is not None
        and result.clause.literals == expected,
        f"status={result.status}, clause={result.clause} (exp {expected})",
    ))
    checks.append(_check(
        "4 depth gate stops before depth three",
        [report.depth for report in result.depth_reports] == [0, 1, 2],
        f"depths={[r.depth for r in result.depth_reports]} (exp [0,1,2])",
    ))
    deeper = Clause(("contact", "target_shape=cross", "target_visible"))
    checks.append(_check(
        "5 MDL prefers the shorter exact rule",
        result.clause is not None
        and mdl_bits(result.clause, trusted(trace), 10) < mdl_bits(deeper, trusted(trace), 10),
        f"minimal={mdl_bits(result.clause, trusted(trace), 10) if result.clause else None} bits, "
        f"deeper={mdl_bits(deeper, trusted(trace), 10)} bits",
    ))

    # No single positive conjunction can express OR: red alone and round alone
    # are positive, while red+round is negative. The grammar must fail loudly.
    gap_trace = [
        Event(frozenset({"red"}), True),
        Event(frozenset({"round"}), True),
        Event(frozenset({"red", "round"}), False),
        Event(frozenset({"blue", "square"}), False),
    ]
    gap = learn_clause(gap_trace, max_depth=2)
    checks.append(_check(
        "6 unexpressible rule reports GRAMMAR_GAP",
        gap.status == "GRAMMAR_GAP" and gap.clause is None,
        f"status={gap.status}, clause={gap.clause} (exp GRAMMAR_GAP,None)",
    ))
    return all(checks)


if __name__ == "__main__":
    print("Offline checks (synthetic confidence-gated event traces):\n")
    if not _run_checks():
        raise SystemExit("\nFix the failed checks before using clause search.")

    result = learn_clause(_training_trace(), max_depth=3)
    print("\nDepth reports:")
    for report in result.depth_reports:
        print(
            f"  depth {report.depth}: {report.candidates:>3} candidates, "
            f"best errors={report.errors}, MDL={report.mdl_bits} bits"
        )
    print(
        f"\n{result.status}: {result.clause}\n"
        f"explored={result.explored_candidates}, "
        f"trusted={result.trusted_events}, filtered={result.filtered_events}\n"
        "The rule names observations, not roles. Next, lesson 0013 will choose "
        "the action that best separates surviving clauses per action spent."
    )
