"""Preconditions a planner search must clear, and the operator audit behind them.

This module's source text is INLINED verbatim into Tycho's
`wm_feedback_probe.py.tmpl` by `wm_probe.py` and executed inside Tycho's
sandbox, where `tycho_nooa` is not installed. It therefore imports nothing
outside the standard library, takes `world_model` and `wmlib` as arguments
rather than importing them, and omits `from __future__ import annotations` --
a `__future__` import is only legal at the top of a file, and this file's text
ends up in the middle of one.

Two questions, in this order:

1. Can `outcome()` return "level_complete" at all? If not, no search over this
   model can succeed -- not with a different `actions()`, `planner_key()`,
   heuristic, or node budget -- and a planner result would report a goal defect
   as a search defect. This is the only hard gate here.
2. What kind of operator is each entry of `actions()`, and does the state graph
   close under them? Reported, never enforced: which operators belong in a
   search alphabet is the builder's call against game evidence, not ours.

Everything measured here goes through wmlib's own `_planner_key`,
`_planner_successor`, and `_acts`. They are private, and deliberately used
anyway: auditing any other projection, copy discipline, or alphabet would
describe a search nobody performs.
"""

import ast
import dataclasses
import textwrap
import time

GOAL_LITERAL = "level_complete"

_CHAIN_DEPTH = 3  # forward states sampled for the per-operator audit
_REPEAT_DEPTH = 8  # repeated applications used to look for a key repeat
_CLOSURE_MAX_EDGES = 400  # bounded exploration used for the closure statistic


def wmgate_report(world_model, wmlib, *, source=None, budget_s=0.75):
    """Return the planner preconditions for `world_model`, never raising.

    Every failure path leaves ``goal_satisfiable`` True: a bug in this gate must
    cost the search nothing, because a false block is invisible in exactly the
    way the unsatisfiable goal was.
    """
    report = {
        "goal_satisfiable": True,
        "goal": {},
        "operators": [],
        "closure": None,
        "closure_forward": None,
        "error": None,
    }
    try:
        report["goal"] = _goal_facts(world_model, wmlib, source)
        report["goal_satisfiable"] = bool(report["goal"].get("satisfiable", True))
        start = wmlib.current_state(world_model)
        if start is None:
            return report
        report["operators"] = _operator_audit(
            world_model, wmlib, start, time.monotonic() + budget_s
        )
        report["closure"] = _closure_facts(
            world_model, wmlib, start, time.monotonic() + budget_s
        )
        forward = {
            entry["label"]
            for entry in report["operators"]
            if entry["role"] == "forward"
        }
        # Only worth a second pass when it would describe a different alphabet: the
        # question it answers is what search costs AFTER the cleanup this recommends.
        if forward and len(forward) < len(report["operators"]):
            report["closure_forward"] = _closure_facts(
                world_model, wmlib, start, time.monotonic() + budget_s, allowed=forward
            )
    except Exception as error:  # noqa: BLE001 - a gate must never break the probe
        report["error"] = f"{type(error).__name__}: {error}"
    return report


def wmgate_blocked_lines(report):
    """Lines replacing a search that provably cannot reach a goal state."""
    goal = report.get("goal") or {}
    lines = [
        "PLANNER_PROBE:",
        "  PLANNER_STATUS: not_run_goal_unsatisfiable",
        '  GOAL_PREDICATE: is_level_complete(s) is outcome(s) == "level_complete"',
    ]
    constant = goal.get("constant_return")
    if constant is not None:
        lines.append(
            "  GOAL_EVIDENCE: outcome() returns the constant "
            f'"{constant}" unconditionally'
        )
    if goal.get("literal_in_module") is False:
        lines.append(
            f'  GOAL_EVIDENCE: the literal "{GOAL_LITERAL}" appears in no '
            "executable string in world_model.py"
        )
    observed = goal.get("observed_values")
    if observed:
        lines.append(
            f"  GOAL_OBSERVED_OUTCOMES: {sorted(observed)} over "
            f"{goal.get('sampled_states', 0)} sampled states (current state and "
            "its one-step successors)"
        )
    lines += [
        (
            "  -> no state in the modelled space satisfies the goal, so search cannot "
            "succeed here: not"
        ),
        (
            "     with a different actions(), planner_key(), heuristic, or node budget. "
            "Running it would"
        ),
        (
            "     report no_plan_within_budget or planner_timeout, and that result would "
            "be about the"
        ),
        "     goal, not about search.",
        (
            "  -> write a candidate terminal predicate in outcome() first. It does not "
            "need terminal"
        ),
        (
            "     evidence to be worth planning against: keep it explicitly a candidate, "
            "and let the"
        ),
        "     verifier falsify it when a level actually ends.",
    ]
    return lines + wmgate_audit_lines(report)


def wmgate_audit_lines(report):
    """Lines describing the declared operator alphabet and the graph it induces."""
    if report.get("error"):
        return ["PLANNER_CONTRACT:", "  PROBE_ERROR: " + str(report["error"])]
    operators = report.get("operators") or []
    lines = _operator_lines(operators) + _closure_lines(report)
    lines += _open_graph_warning(report)
    return ["PLANNER_CONTRACT:"] + lines if lines else []


def _operator_lines(operators):
    if not operators:
        return []
    counts = {}
    for entry in operators:
        counts[entry["role"]] = counts.get(entry["role"], 0) + 1
    summary = ", ".join(f"{n} {role}" for role, n in sorted(counts.items()))
    lines = [f"  PLANNER_OPERATORS: {len(operators)} declared by actions(); {summary}"]
    for entry in operators:
        head = f"    {entry['label']:<11}{entry['role']:<20}"
        lines += textwrap.wrap(
            entry["note"],
            width=88,
            initial_indent=head,
            subsequent_indent=" " * len(head),
        )
    if any(entry["role"] in _NON_FORWARD for entry in operators):
        lines += [
            (
                "  -> a search alphabet wants forward operators only. Non-forward "
                "entries multiply the"
            ),
            (
                "     branching factor without reaching any state a forward path "
                "cannot reach."
            ),
        ]
    return lines


def _closure_lines(report):
    return [
        (
            f"  PLANNER_STATE_GRAPH: {caption}: edges={closure['edges']} "
            f"distinct_keys={closure['distinct_keys']} "
            f"revisits={closure['revisits']} -> "
            + ("CLOSING" if closure["revisits"] else "NOT CLOSING")
        )
        for closure, caption in _closures(report)
    ]


def _open_graph_warning(report):
    # The forward-only figure, when present, is the one that matters: it is the
    # graph the search would explore after taking the cleanup above.
    decisive = (report.get("closure_forward") or report.get("closure")) or {}
    if not decisive.get("edges") or decisive.get("revisits"):
        return []
    return [
        (
            "  -> planner_key() merged no two states in that sample, so search "
            "explores a tree:"
        ),
        ("     cost is exponential in plan length and no node budget fixes it. Define"),
        (
            "     planner_key(state) carrying exactly the state those operators "
            "read -- it must be"
        ),
        (
            "     congruent with that alphabet AND with the goal predicate, which "
            "are two separate"
        ),
        (
            "     obligations, and a key congruent with only one of them makes the "
            "planner fast and"
        ),
        "     wrong rather than slow and right.",
    ]


def _closures(report):
    """(statistic, caption) pairs, widest alphabet first."""
    pairs = []
    total = len(report.get("operators") or [])
    if report.get("closure"):
        pairs.append((report["closure"], f"all {total} declared operators"))
    if report.get("closure_forward"):
        pairs.append(
            (
                report["closure_forward"],
                f"the {report['closure_forward']['operators']} forward operators only",
            )
        )
    return pairs


# ---- goal satisfiability -------------------------------------------------


def _goal_facts(world_model, wmlib, source):
    """Decide whether `outcome()` can return "level_complete" at all.

    Two independently sufficient static conditions, either of which settles it:

    - `outcome()` is one unconditional `return <string>` that is not the goal
      literal. Nothing else in the module can change what it returns.
    - the goal literal appears in no executable string anywhere in the module,
      so no function can return it short of assembling it at runtime.

    The second alone is not enough, and a live run proved it: pinned Tycho's
    seeded world_model.py already spends the literal in `subgoals()` --
    `return [lambda s: outcome(s) == "level_complete"]` -- while `outcome()` is
    still `return "ongoing"`. The module-wide search sees the literal and clears
    a model that provably has no goal state, which is the exact model this gate
    exists to catch. See tests/test_probe_gate.py::test_the_seed_world_model_is_blocked.

    The dynamic half is corroboration only: sampling states can exhibit a goal
    but can never prove there is none.
    """
    facts = {
        "satisfiable": True,
        "literal_in_module": None,
        "constant_return": None,
        "observed_values": [],
        "sampled_states": 0,
    }
    text = source if source is not None else _module_source(world_model)
    if text is not None:
        tree = ast.parse(text)
        facts["literal_in_module"] = GOAL_LITERAL in _executable_strings(tree)
        facts["constant_return"] = _constant_return(tree, "outcome")
        constant = facts["constant_return"]
        facts["satisfiable"] = not (
            (constant is not None and constant != GOAL_LITERAL)
            or facts["literal_in_module"] is False
        )
    try:
        states = _sample_states(world_model, wmlib)
        facts["sampled_states"] = len(states)
        facts["observed_values"] = sorted(
            {str(wmlib.outcome(world_model, s)) for s in states}
        )
    except Exception:  # noqa: BLE001, S110 - corroboration only, never the verdict
        pass
    return facts


def _executable_strings(tree):
    """String constants the module can actually return, i.e. minus docstrings."""
    docstrings = set()
    for node in ast.walk(tree):
        if (docstring_id := _docstring_id(node)) is not None:
            docstrings.add(docstring_id)
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    }


def _docstring_id(node):
    owners = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    if not isinstance(node, owners):
        return None
    body = getattr(node, "body", None)
    if not body or not isinstance(body[0], ast.Expr):
        return None
    first = body[0].value
    if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
        return None
    return id(first)


def _constant_return(tree, name):
    """The single string a top-level function returns unconditionally, else None."""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return _unconditional_string_return(node)
    return None


def _unconditional_string_return(function):
    body = list(function.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return None
    value = body[0].value
    if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
        return None
    return value.value


def _module_source(world_model):
    path = getattr(world_model, "__file__", None)
    if not path:
        return None
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return None


def _sample_states(world_model, wmlib):
    start = wmlib.current_state(world_model)
    if start is None:
        return []
    states = [start]
    for action in _actions(world_model, wmlib, start):
        try:
            states.append(_successor(world_model, wmlib, start, action))
        except Exception:  # noqa: BLE001, S110 - a broken operator is the audit's job
            pass
    return states


# ---- operator alphabet ---------------------------------------------------


_NON_FORWARD = ("no_op", "restart", "retrograde", "retrograde_partial")

_ROLE_NOTES = {
    "no_op": "successor key equals its own key from every sampled state",
    "restart": "successor key equals this level's init_state key",
    "retrograde": (
        "reaches a key already behind it on the sampled path; its successor reads "
        "state the planner key must then carry"
    ),
}


def _operator_audit(world_model, wmlib, start, deadline):
    """Classify each declared action by what it does to the planner's own key.

    The classification is behavioural rather than name-based: an action named
    UNDO that does not undo, or a game where RESET is the only way forward, has
    to come out of this by what it does.
    """
    chain, chain_keys = _forward_chain(world_model, wmlib, start)
    init_key = _init_key(world_model, wmlib)
    views = _varying_views(wmlib, chain)
    entries = []
    for action in _actions(world_model, wmlib, start):
        if time.monotonic() > deadline:
            break
        entries.append(
            _classify(world_model, wmlib, action, chain, chain_keys, init_key, views)
        )
    return entries


def _classify(world_model, wmlib, action, chain, chain_keys, init_key, views):
    label = _label(action)
    try:
        successors = [_successor(world_model, wmlib, s, action) for s in chain]
    except Exception as error:  # noqa: BLE001
        return {
            "label": label,
            "role": "error",
            "note": f"transition() raised {type(error).__name__}: {error}",
        }
    keys = [_key(world_model, wmlib, s) for s in successors]
    role = _role(keys, chain_keys, init_key)
    if role is not None:
        return {"label": label, "role": role, "note": _ROLE_NOTES[role]}
    partial = _partial_return(wmlib, views, successors)
    if partial is not None:
        return {"label": label, "role": "retrograde_partial", "note": partial}
    repeats = _repeats(world_model, wmlib, chain[0], action)
    note = (
        f"key repeats after {repeats} repeated applications"
        if repeats is not None
        else f"no key repeat within {_REPEAT_DEPTH} repeated applications"
    )
    return {"label": label, "role": "forward", "note": note}


def _role(keys, chain_keys, init_key):
    """The key-level role of an operator, or None when it is plainly forward."""
    if all(after == before for after, before in zip(keys, chain_keys)):
        return "no_op"
    if init_key is not None and any(
        after == init_key and before != init_key
        for after, before in zip(keys, chain_keys)
    ):
        return "restart"
    if any(after in chain_keys[:index] for index, after in enumerate(keys)):
        return "retrograde"
    return None


def _partial_return(wmlib, views, successors):
    """Name the fields that stop a backward-moving operator from closing the graph.

    An operator can restore every field the game shows and still produce a fresh
    key, because one field it does not restore rides along -- a move counter, a
    history log. That operator is retrograde in the game and forward in the
    search, which is the worst of both: it doubles the branching factor and can
    never revisit. Naming the blocking field is the whole diagnosis, so this is
    reported rather than judged: the fields are printed and the builder decides.
    """
    if not views:
        return None
    names = sorted(views[0])
    for index, state in enumerate(successors):
        after = _field_view(wmlib, state, names)
        if after is None:
            continue
        match = _partial_ancestor(names, after, views[:index])
        if match is not None:
            offset, same, differ = match
            return (
                f"returns {len(same)} of {len(names)} varying state fields to a "
                f"state {_plural(index - offset, 'step')} back; "
                f"{_follow_clause(differ)}, so the key never closes"
            )
    return None


def _partial_ancestor(names, after, ancestors):
    for offset, ancestor in enumerate(ancestors):
        same = [name for name in names if after.get(name) == ancestor.get(name)]
        differ = [name for name in names if name not in same]
        if same and differ and len(differ) < len(same):
            return offset, same, differ
    return None


def _plural(count, noun):
    return f"{count} {noun}" + ("" if count == 1 else "s")


def _follow_clause(fields):
    return "{} {} not follow".format(
        ", ".join(fields),
        "does" if len(fields) == 1 else "do",
    )


def _varying_views(wmlib, chain):
    """Field views of the sampled chain, restricted to fields that actually vary.

    Immutable terrain and shape tables match between any two states and would
    drown the comparison; only the fields the dynamics move carry information
    about what an operator did.
    """
    views = [_field_view(wmlib, state, None) for state in chain]
    if len(views) < 2 or any(view is None for view in views):
        return []
    varying = _varying_names(views)
    if not varying:
        return []
    return [{name: view[name] for name in varying} for view in views]


def _varying_names(views):
    return [
        name
        for name in views[0]
        if any(view.get(name) != views[0][name] for view in views[1:])
    ]


def _field_view(wmlib, state, names):
    """A hashable per-field view of a state, or None when it has no field structure."""
    raw = _raw_fields(state)
    if raw is None:
        return None
    selected = (
        raw if names is None else {name: raw[name] for name in names if name in raw}
    )
    try:
        return {name: wmlib._hashable(value) for name, value in selected.items()}
    except Exception:  # noqa: BLE001
        return None


def _raw_fields(state):
    if dataclasses.is_dataclass(state) and not isinstance(state, type):
        return {
            field.name: getattr(state, field.name)
            for field in dataclasses.fields(state)
        }
    if hasattr(state, "_asdict"):
        return dict(state._asdict())
    if hasattr(state, "__dict__") and vars(state):
        return dict(vars(state))
    return None


def _repeats(world_model, wmlib, start, action):
    """Applications after which repeating `action` revisits a key, else None.

    An operator that never repeats a key under repetition is unbounded in the
    planner's own state space -- an unbounded counter field, or a position the
    model lets walk off the grid -- and each one multiplies the search.
    """
    state = start
    seen = {_key(world_model, wmlib, state)}
    for step in range(1, _REPEAT_DEPTH + 1):
        try:
            state = _successor(world_model, wmlib, state, action)
        except Exception:  # noqa: BLE001
            return None
        current = _key(world_model, wmlib, state)
        if current in seen:
            return step
        seen.add(current)
    return None


def _forward_chain(world_model, wmlib, start):
    """A short path of key-distinct states, used as the audit's sample.

    One state is not enough: an operator can be a no-op at the start state by
    coincidence, and a retrograde operator has nothing to go back to there.
    """
    chain = [start]
    keys = [_key(world_model, wmlib, start)]
    for _ in range(_CHAIN_DEPTH):
        for action in _actions(world_model, wmlib, chain[-1]):
            try:
                nxt = _successor(world_model, wmlib, chain[-1], action)
            except Exception:  # noqa: BLE001, S112 - skip the edge, keep the sample
                continue
            candidate = _key(world_model, wmlib, nxt)
            if candidate not in keys:
                chain.append(nxt)
                keys.append(candidate)
                break
        else:
            break
    return chain, keys


def _init_key(world_model, wmlib):
    """This level's initial-state key, so restart-like operators are visible."""
    try:
        level, _ = wmlib.current_frame_key()
        frames = wmlib.frames()
        turn = min(t for (lvl, t) in frames if lvl == level)
        state = world_model.init_state(frames[(level, turn)], level)
        return _key(world_model, wmlib, state)
    except Exception:  # noqa: BLE001
        return None


# ---- graph closure -------------------------------------------------------


def _closure_facts(world_model, wmlib, start, deadline, allowed=None):
    """Does the key ever merge two search paths, within a bounded exploration?

    A revisit count of zero is the fact behind an exponential search: it says the
    model's state graph, as the planner sees it, is a tree.
    """
    frontier = [start]
    seen = {_key(world_model, wmlib, start)}
    edges = 0
    revisits = 0
    while _closure_open(frontier, edges, deadline):
        state = frontier.pop(0)
        for action in _actions(world_model, wmlib, state):
            if edges >= _CLOSURE_MAX_EDGES:
                break
            if allowed is not None and _label(action) not in allowed:
                continue
            succeeded, nxt = _closure_successor(world_model, wmlib, state, action)
            if not succeeded:
                continue
            edges += 1
            candidate = _key(world_model, wmlib, nxt)
            if candidate in seen:
                revisits += 1
            else:
                seen.add(candidate)
                frontier.append(nxt)
    return {
        "operators": len(allowed) if allowed is not None else None,
        "edges": edges,
        "distinct_keys": len(seen),
        "revisits": revisits,
        "truncated": bool(frontier),
    }


def _closure_open(frontier, edges, deadline):
    return (
        bool(frontier) and edges < _CLOSURE_MAX_EDGES and time.monotonic() <= deadline
    )


def _closure_successor(world_model, wmlib, state, action):
    try:
        return True, _successor(world_model, wmlib, state, action)
    except Exception:  # noqa: BLE001 - skip the edge, keep the bounded sample
        return False, None


# ---- the planner's own primitives ----------------------------------------


def _key(world_model, wmlib, state):
    hook = getattr(wmlib, "_planner_key", None)
    return hook(world_model, state) if hook is not None else repr(state)


def _successor(world_model, wmlib, state, action):
    hook = getattr(wmlib, "_planner_successor", None)
    if hook is not None:
        return hook(world_model, state, action)
    return world_model.transition(wmlib.clone(state), action)


def _actions(world_model, wmlib, state):
    hook = getattr(wmlib, "_acts", None)
    if hook is not None:
        return hook(world_model, state)
    return list(world_model.actions(state))


def _label(action):
    if isinstance(action, dict):
        name = str(action.get("action"))
        row, col = action.get("row"), action.get("col")
        return name if row is None and col is None else f"{name}({row},{col})"
    return str(action)
