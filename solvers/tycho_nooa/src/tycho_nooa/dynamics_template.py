"""Bounded diagnostics for candidate transition systems.

This file is seeded into each fresh workspace as ``dynamics.py``.  It is
machinery, not game judgment: the caller supplies the candidate model, states,
actions, and any projection that says which state distinctions matter.

``key_counterexample`` searches breadth-first for the shortest tested action
suffix that refutes a proposed planner-key equivalence.  Not finding one never
proves equivalence; the result remains ``inconclusive`` and records the bound.

``action_laws`` checks idempotence, directional inverse laws, commutation, and
many-to-one behaviour on exactly the supplied finite state sample.  A law marked
``supported`` is supported only on that declared domain and representation.
Every returned certificate records its representation, exclusions, evidence
identifiers, and resource cap so a bounded result cannot masquerade as a theorem.
Callable provenance follows referenced globals for two levels and records that
depth; canonical per-check evidence, not the commitment hash alone, supports the
finite-domain judgment.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import marshal
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from itertools import combinations, product
from pathlib import Path
from types import CodeType, ModuleType
from typing import Any

State = Any
Action = Any
Projection = Callable[[State], Any]


def key_counterexample(
    model: Any,
    states: Sequence[State],
    *,
    actions: Iterable[Action] | None = None,
    key: Projection | None = None,
    observe: Projection | None = None,
    representation: str | None = None,
    max_depth: int = 4,
    max_nodes: int = 1_000,
    max_actions: int = 16,
    inference_transition_ids: Iterable[str] = (),
    held_out_transition_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Return the shortest bounded suffix that distinguishes two key-equal states.

    ``key`` defaults to ``model.planner_key``. ``observe`` defaults to the
    candidate model's render, outcome, and available-action signature.  Supply a
    projection explicitly when the planning representation intentionally forgets
    some rendered or latent fields, and name every custom observation with
    ``representation`` so its certificate is not anonymous.
    """
    left, right = _state_pair(states)
    _positive_bound("max_nodes", max_nodes)
    _positive_bound("max_actions", max_actions)
    if max_depth < 0:
        raise ValueError("max_depth must be at least zero")
    key_fn: Projection = (
        key if key is not None else getattr(model, "planner_key", _identity)
    )
    observation: Projection = (
        observe
        if observe is not None
        else lambda state: _model_observation(model, state)
    )
    representation_name = _representation_name(
        representation,
        custom=observe is not None,
        default="model.render+outcome+actions",
    )
    fixed_actions = None if actions is None else _ordered(actions)
    _validate_action_cap(fixed_actions, max_actions)
    representation_details = _representation_details(
        (left, right),
        observation,
        representation_name,
        commitment=(
            None
            if observe is not None
            else _named_commitment("model.render+outcome+actions")
        ),
    )
    left_key = _freeze(key_fn(copy.deepcopy(left)))
    right_key = _freeze(key_fn(copy.deepcopy(right)))
    certificate = _key_certificate(
        representation_name,
        representation_details,
        _callable_commitment(key_fn),
        _callable_commitment(model.transition, global_depth=0),
        (_value_hash(left_key), _value_hash(right_key)),
        fixed_actions,
        max_depth,
        max_nodes,
        max_actions,
        inference_transition_ids,
        held_out_transition_ids,
    )
    if left_key != right_key:
        certificate["exclusions"].append("initial states have different planner keys")
        return certificate

    return _search_key_counterexample(
        model,
        left,
        right,
        fixed_actions,
        observation,
        max_depth,
        max_nodes,
        max_actions,
        certificate,
    )


def action_laws(
    model: Any,
    states: Sequence[State],
    *,
    actions: Iterable[Action] | None = None,
    projection: Projection | None = None,
    representation: str | None = None,
    max_checks: int = 1_000,
    max_actions: int = 16,
    inference_transition_ids: Iterable[str] = (),
    held_out_transition_ids: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Check small action-algebra claims on a finite, caller-chosen domain.

    A custom ``projection`` requires a stable, caller-chosen ``representation``
    name. Certificates bind that name to the projection implementation, preserve
    the canonical finite domain, and record every successful comparison used for
    a universal finite-domain judgment.
    """
    if not states:
        raise ValueError("states must contain at least one state")
    _positive_bound("max_checks", max_checks)
    project = projection or _identity
    representation_name = _representation_name(
        representation, custom=projection is not None, default="full_state"
    )
    ordered_actions = _ordered(
        actions if actions is not None else _all_actions(model, states)
    )
    if not ordered_actions:
        raise ValueError("actions must contain at least one action")
    context = _law_context(
        model,
        states,
        ordered_actions,
        project,
        representation_name,
        max_checks,
        max_actions,
        inference_transition_ids,
        held_out_transition_ids,
    )
    certificates = [
        _universal_law(
            model, context, "idempotent", (action,), (action, action), (action,)
        )
        for action in ordered_actions
    ]
    certificates.extend(
        _universal_law(model, context, "inverse", pair, pair, ())
        for pair in product(ordered_actions, repeat=2)
    )
    certificates.extend(
        _universal_law(
            model,
            context,
            "commutes",
            pair,
            pair,
            tuple(reversed(pair)),
        )
        for pair in combinations(ordered_actions, 2)
    )
    certificates.extend(
        _many_to_one_law(model, context, action) for action in ordered_actions
    )
    return certificates


@dataclass(frozen=True)
class _LawContext:
    states: tuple[State, ...]
    actions: tuple[Action, ...]
    projection: Projection
    representation: str
    representation_hash: str
    projection_commitment: dict[str, Any]
    theory_commitment: dict[str, Any]
    state_values: tuple[Any, ...]
    projected_state_values: tuple[Any, ...]
    state_hashes: tuple[str, ...]
    projected_state_hashes: tuple[str, ...]
    max_checks: int
    max_actions: int
    inference_transition_ids: tuple[str, ...]
    held_out_transition_ids: tuple[str, ...]


def _key_certificate(
    representation: str,
    representation_details: dict[str, Any],
    key_commitment: dict[str, Any],
    theory_commitment: dict[str, Any],
    key_hashes: tuple[str, str],
    actions: tuple[Action, ...] | None,
    max_depth: int,
    max_nodes: int,
    max_actions: int,
    inference_transition_ids: Iterable[str],
    held_out_transition_ids: Iterable[str],
) -> dict[str, Any]:
    return {
        "kind": "key_counterexample",
        "status": "inconclusive",
        "claim": "planner-key-equal states have indistinguishable tested futures",
        "representation": representation,
        "representation_hash": representation_details["representation_hash"],
        "projection_commitment": representation_details["projection_commitment"],
        "key_commitment": key_commitment,
        "theory_commitment": theory_commitment,
        "domain": {
            "actions": (
                "model.actions"
                if actions is None
                else [_freeze(action) for action in actions]
            ),
            "key_hashes": list(key_hashes),
            "max_depth": max_depth,
            "projected_state_values": representation_details["projected_state_values"],
            "projected_state_hashes": representation_details["projected_state_hashes"],
            "state_values": representation_details["state_values"],
            "state_hashes": representation_details["state_hashes"],
        },
        "inference_transition_ids": list(inference_transition_ids),
        "held_out_transition_ids": list(held_out_transition_ids),
        "counterexample": None,
        "exclusions": [],
        "resource_cap": {
            "max_actions": max_actions,
            "max_nodes": max_nodes,
            "nodes_checked": 0,
            "depth_exhausted": False,
            "node_cap_exhausted": False,
        },
    }


def _search_key_counterexample(
    model: Any,
    left: State,
    right: State,
    fixed_actions: tuple[Action, ...] | None,
    observation: Projection,
    max_depth: int,
    max_nodes: int,
    max_actions: int,
    certificate: dict[str, Any],
) -> dict[str, Any]:
    frontier: deque[tuple[State, State, tuple[Action, ...]]] = deque(
        [(left, right, ())]
    )
    depth_exhausted = False
    while frontier:
        if certificate["resource_cap"]["nodes_checked"] == max_nodes:
            break
        layer_size = len(frontier)
        pending_counterexample = None
        for _ in range(layer_size):
            if certificate["resource_cap"]["nodes_checked"] == max_nodes:
                certificate["resource_cap"]["depth_exhausted"] = depth_exhausted
                certificate["resource_cap"]["node_cap_exhausted"] = True
                return certificate
            left_state, right_state, suffix = frontier.popleft()
            certificate["resource_cap"]["nodes_checked"] += 1
            counterexample = _observation_counterexample(
                observation, left_state, right_state, suffix
            )
            if counterexample is not None:
                return _contradicted(certificate, counterexample)
            if len(suffix) == max_depth:
                depth_exhausted = True
                continue
            candidate = _extend_key_frontier(
                frontier,
                model,
                left_state,
                right_state,
                suffix,
                fixed_actions,
                max_actions,
            )
            pending_counterexample = pending_counterexample or candidate
        if pending_counterexample is not None:
            return _contradicted(certificate, pending_counterexample)
    certificate["resource_cap"]["depth_exhausted"] = depth_exhausted
    certificate["resource_cap"]["node_cap_exhausted"] = bool(frontier)
    return certificate


def _observation_counterexample(
    observe: Projection,
    left: State,
    right: State,
    suffix: tuple[Action, ...],
) -> dict[str, Any] | None:
    left_observation = _freeze(observe(copy.deepcopy(left)))
    right_observation = _freeze(observe(copy.deepcopy(right)))
    if left_observation == right_observation:
        return None
    return {
        "left_observation": left_observation,
        "right_observation": right_observation,
        "suffix": [_freeze(action) for action in suffix],
    }


def _extend_key_frontier(
    frontier: deque[tuple[State, State, tuple[Action, ...]]],
    model: Any,
    left: State,
    right: State,
    suffix: tuple[Action, ...],
    fixed_actions: tuple[Action, ...] | None,
    max_actions: int,
) -> dict[str, Any] | None:
    candidates = (
        fixed_actions
        if fixed_actions is not None
        else _shared_actions(model, left, right)
    )
    _validate_action_cap(candidates, max_actions)
    for action in candidates:
        successors = _paired_successors(model, left, right, action)
        if successors["separates"]:
            return {
                "left_observation": successors["left"],
                "right_observation": successors["right"],
                "suffix": [*[_freeze(item) for item in suffix], _freeze(action)],
            }
        if successors["states"] is not None:
            next_left, next_right = successors["states"]
            frontier.append((next_left, next_right, (*suffix, action)))
    return None


def _contradicted(
    certificate: dict[str, Any], counterexample: dict[str, Any]
) -> dict[str, Any]:
    certificate["status"] = "contradicted"
    certificate["counterexample"] = counterexample
    return certificate


def _law_context(
    model: Any,
    states: Sequence[State],
    actions: tuple[Action, ...],
    projection: Projection,
    representation: str,
    max_checks: int,
    max_actions: int,
    inference_transition_ids: Iterable[str],
    held_out_transition_ids: Iterable[str],
) -> _LawContext:
    _positive_bound("max_actions", max_actions)
    _validate_action_cap(actions, max_actions)
    details = _representation_details(states, projection, representation)
    return _LawContext(
        states=tuple(states),
        actions=actions,
        projection=projection,
        representation=representation,
        representation_hash=details["representation_hash"],
        projection_commitment=details["projection_commitment"],
        theory_commitment=_callable_commitment(model.transition, global_depth=0),
        state_values=tuple(details["state_values"]),
        projected_state_values=tuple(details["projected_state_values"]),
        state_hashes=tuple(details["state_hashes"]),
        projected_state_hashes=tuple(details["projected_state_hashes"]),
        max_checks=max_checks,
        max_actions=max_actions,
        inference_transition_ids=tuple(inference_transition_ids),
        held_out_transition_ids=tuple(held_out_transition_ids),
    )


def _universal_law(
    model: Any,
    context: _LawContext,
    name: str,
    actions: tuple[Action, ...],
    left_path: tuple[Action, ...],
    right_path: tuple[Action, ...],
) -> dict[str, Any]:
    certificate = _law_certificate(context, name, actions)
    project = context.projection
    for state in context.states:
        if certificate["resource_cap"]["checks"] == context.max_checks:
            certificate["resource_cap"]["truncated"] = True
            break
        certificate["resource_cap"]["checks"] += 1
        try:
            left = _freeze(project(_follow(model, state, left_path)))
            right = _freeze(project(_follow(model, state, right_path)))
        except Exception as error:  # noqa: BLE001 - candidate model defines its domain
            certificate["exclusions"].append(_failure(error, state))
            continue
        if left != right:
            certificate["status"] = "contradicted"
            certificate["counterexample"] = {
                "left": left,
                "right": right,
                "state": _freeze(state),
            }
            return certificate
        certificate["evidence"].append(
            {"state_hash": _value_hash(_freeze(state)), "left": left, "right": right}
        )
    if not certificate["exclusions"] and not certificate["resource_cap"]["truncated"]:
        certificate["status"] = "supported"
    return certificate


def _many_to_one_law(
    model: Any, context: _LawContext, action: Action
) -> dict[str, Any]:
    certificate = _law_certificate(context, "many_to_one", (action,))
    project = context.projection
    comparable = 0
    for left_state, right_state in combinations(context.states, 2):
        if certificate["resource_cap"]["checks"] == context.max_checks:
            certificate["resource_cap"]["truncated"] = True
            break
        if _project(project, left_state) == _project(project, right_state):
            continue
        comparable += 1
        certificate["resource_cap"]["checks"] += 1
        try:
            left = _project(project, _follow(model, left_state, (action,)))
            right = _project(project, _follow(model, right_state, (action,)))
        except Exception as error:  # noqa: BLE001 - candidate model defines its domain
            certificate["exclusions"].append(_failure(error, (left_state, right_state)))
            continue
        if left == right:
            certificate["status"] = "supported"
            certificate["witness"] = {
                "left_state": _freeze(left_state),
                "right_state": _freeze(right_state),
                "result": left,
            }
            certificate["evidence"].append(
                {
                    "left_state_hash": _value_hash(_freeze(left_state)),
                    "right_state_hash": _value_hash(_freeze(right_state)),
                    "left": left,
                    "right": right,
                }
            )
            return certificate
        certificate["evidence"].append(
            {
                "left_state_hash": _value_hash(_freeze(left_state)),
                "right_state_hash": _value_hash(_freeze(right_state)),
                "left": left,
                "right": right,
            }
        )
    if comparable == 0:
        certificate["exclusions"].append("fewer than two distinct projected states")
    elif not certificate["exclusions"] and not certificate["resource_cap"]["truncated"]:
        certificate["status"] = "contradicted"
    return certificate


def _law_certificate(
    context: _LawContext, name: str, actions: tuple[Action, ...]
) -> dict[str, Any]:
    frozen_actions = [_freeze(action) for action in actions]
    return {
        "kind": "action_law",
        "law": name,
        "actions": frozen_actions,
        "equation": _law_equation(name, frozen_actions),
        "status": "inconclusive",
        "representation": context.representation,
        "representation_hash": context.representation_hash,
        "projection_commitment": context.projection_commitment,
        "theory_commitment": context.theory_commitment,
        "domain": {
            "actions": [_freeze(action) for action in context.actions],
            "projected_state_values": list(context.projected_state_values),
            "projected_state_hashes": list(context.projected_state_hashes),
            "state_values": list(context.state_values),
            "state_hashes": list(context.state_hashes),
        },
        "inference_transition_ids": list(context.inference_transition_ids),
        "held_out_transition_ids": list(context.held_out_transition_ids),
        "counterexample": None,
        "witness": None,
        "evidence": [],
        "exclusions": [],
        "resource_cap": {
            "max_checks": context.max_checks,
            "max_actions": context.max_actions,
            "checks": 0,
            "truncated": False,
        },
    }


def _law_equation(name: str, actions: list[Any]) -> str:
    labels = [_canonical(action) for action in actions]
    if name == "idempotent":
        return f"T[{labels[0]}](T[{labels[0]}](s)) = T[{labels[0]}](s)"
    if name == "inverse":
        return f"T[{labels[1]}](T[{labels[0]}](s)) = s"
    if name == "commutes":
        return f"T[{labels[1]}](T[{labels[0]}](s)) = T[{labels[0]}](T[{labels[1]}](s))"
    return f"exists s != t: T[{labels[0]}](s) = T[{labels[0]}](t)"


def _paired_successors(
    model: Any, left: State, right: State, action: Action
) -> dict[str, Any]:
    left_result = _successor_or_failure(model, left, action)
    right_result = _successor_or_failure(model, right, action)
    if left_result[0] or right_result[0]:
        left_failure = left_result[1] if left_result[0] else {"transition": "succeeded"}
        right_failure = (
            right_result[1] if right_result[0] else {"transition": "succeeded"}
        )
        return {
            "separates": left_failure != right_failure,
            "left": left_failure,
            "right": right_failure,
            "states": None,
        }
    return {
        "separates": False,
        "left": None,
        "right": None,
        "states": (left_result[1], right_result[1]),
    }


def _successor_or_failure(model: Any, state: State, action: Action) -> tuple[bool, Any]:
    try:
        return False, model.transition(copy.deepcopy(state), copy.deepcopy(action))
    except Exception as error:  # noqa: BLE001 - a model failure is certificate evidence
        return True, _error(error)


def _follow(model: Any, state: State, actions: Iterable[Action]) -> State:
    current = copy.deepcopy(state)
    for action in actions:
        current = model.transition(current, copy.deepcopy(action))
    return current


def _model_observation(model: Any, state: State) -> dict[str, Any]:
    render = getattr(model, "render", _identity)
    outcome = getattr(model, "outcome", None)
    return {
        "render": _freeze(render(copy.deepcopy(state))),
        "outcome": _freeze(outcome(copy.deepcopy(state))) if outcome else None,
        "actions": [_freeze(action) for action in _all_actions(model, (state,))],
    }


def _shared_actions(model: Any, left: State, right: State) -> tuple[Action, ...]:
    return _ordered((*_actions(model, left), *_actions(model, right)))


def _all_actions(model: Any, states: Sequence[State]) -> tuple[Action, ...]:
    return _ordered(action for state in states for action in _actions(model, state))


def _actions(model: Any, state: State) -> tuple[Action, ...]:
    available = getattr(model, "actions", None)
    return tuple(available(copy.deepcopy(state))) if available else ()


def _ordered(actions: Iterable[Action]) -> tuple[Action, ...]:
    unique: dict[str, Action] = {}
    for action in actions:
        unique.setdefault(_canonical(_freeze(action)), action)
    return tuple(unique[key] for key in sorted(unique))


def _representation_details(
    states: Sequence[State],
    projection: Projection,
    representation: str,
    commitment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state_values = [_freeze(state) for state in states]
    projected_state_values = [_project(projection, state) for state in states]
    projection_commitment = commitment or _callable_commitment(projection)
    return {
        "representation_hash": _value_hash(
            {
                "name": representation,
                "projection_identity": projection_commitment["identity_sha256"],
                "projected_state_values": projected_state_values,
            }
        ),
        "projection_commitment": projection_commitment,
        "projected_state_values": projected_state_values,
        "projected_state_hashes": [
            _value_hash(value) for value in projected_state_values
        ],
        "state_values": state_values,
        "state_hashes": [_value_hash(value) for value in state_values],
    }


def _named_commitment(name: str) -> dict[str, Any]:
    source_path = __file__
    identity = {"builtin": name, "source_sha256": _source_hash(source_path)}
    return identity | {
        "identity_sha256": _value_hash(identity),
        "source_path": source_path,
    }


def _callable_commitment(
    function: Callable[..., Any],
    seen: frozenset[int] = frozenset(),
    *,
    global_depth: int = 2,
) -> dict[str, Any]:
    logical_name = _logical_name(function)
    if id(function) in seen:
        return {"cycle": logical_name, "identity_sha256": _value_hash(logical_name)}
    seen = seen | {id(function)}
    code = getattr(function, "__code__", None)
    source_target = (
        function
        if inspect.isroutine(function) or inspect.isclass(function)
        else type(function)
    )
    source_path = _source_path(source_target)
    identity = {
        "module": getattr(function, "__module__", type(function).__module__),
        "qualname": getattr(function, "__qualname__", type(function).__qualname__),
        "code_sha256": _code_hash(code),
        "global_depth": global_depth,
        "defaults": _commitment_value(
            getattr(function, "__defaults__", None), seen, global_depth
        ),
        "keyword_defaults": _commitment_value(
            getattr(function, "__kwdefaults__", None), seen, global_depth
        ),
        "closure": _closure_values(function, code, seen, global_depth),
        "globals": (
            _referenced_globals(function, code, seen, global_depth)
            if global_depth > 0
            else {}
        ),
        "bound_state": _object_state(
            getattr(function, "__self__", None), seen, global_depth
        ),
        "callable_state": _object_state(function, seen, global_depth),
        "source_sha256": _source_hash(source_path),
    }
    return identity | {
        "identity_sha256": _value_hash(identity),
        "source_path": source_path,
    }


def _closure_values(
    function: Callable[..., Any],
    code: CodeType | None,
    seen: frozenset[int],
    global_depth: int,
) -> list[Any]:
    closure = getattr(function, "__closure__", None)
    if not isinstance(closure, tuple):
        return []
    names = () if not isinstance(code, CodeType) else code.co_names
    return [
        _contextual_value(cell.cell_contents, names, seen, global_depth)
        for cell in closure
    ]


def _referenced_globals(
    function: Callable[..., Any],
    code: CodeType | None,
    seen: frozenset[int],
    global_depth: int,
) -> dict[str, Any]:
    namespace = getattr(function, "__globals__", {})
    if not isinstance(code, CodeType) or not isinstance(namespace, Mapping):
        return {}
    names = code.co_names
    return {
        name: _contextual_value(namespace[name], names, seen, global_depth)
        for name in sorted(set(names))
        if name in namespace and name != "__builtins__"
    }


def _contextual_value(
    value: Any,
    names: tuple[str, ...],
    seen: frozenset[int],
    global_depth: int,
) -> Any:
    if not isinstance(value, ModuleType):
        return _commitment_value(value, seen, global_depth)
    path = _source_path(value)
    attributes = {
        name: _commitment_value(vars(value)[name], seen | {id(value)}, global_depth)
        for name in sorted(set(names))
        if name in vars(value) and name != "__builtins__"
    }
    return {
        "module": value.__name__,
        "source_sha256": _source_hash(path),
        "referenced_attributes": attributes,
    }


def _object_state(value: Any, seen: frozenset[int], global_depth: int) -> Any:
    try:
        state = vars(value)
    except TypeError:
        return None
    return {
        key: _commitment_value(item, seen, global_depth)
        for key, item in sorted(state.items())
        if not key.startswith("__")
    }


def _commitment_value(value: Any, seen: frozenset[int], global_depth: int = 0) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if id(value) in seen:
        return {"cycle": _logical_name(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, ModuleType):
        path = _source_path(value)
        return {"module": value.__name__, "source_sha256": _source_hash(path)}
    if callable(value):
        return {
            "callable_identity": _callable_commitment(
                value, seen, global_depth=max(0, global_depth - 1)
            )["identity_sha256"]
        }
    return _structured_commitment(value, seen | {id(value)}, global_depth)


def _structured_commitment(value: Any, seen: frozenset[int], global_depth: int) -> Any:
    if isinstance(value, Mapping):
        return _mapping_commitment(value, seen, global_depth)
    if isinstance(value, (list, tuple)):
        return [_commitment_value(item, seen, global_depth) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(
            (_commitment_value(item, seen, global_depth) for item in value),
            key=_canonical,
        )
    return _object_commitment(value, seen, global_depth)


def _mapping_commitment(
    value: Mapping[Any, Any], seen: frozenset[int], global_depth: int
) -> Any:
    pairs = [
        [
            _commitment_value(key, seen, global_depth),
            _commitment_value(item, seen, global_depth),
        ]
        for key, item in value.items()
    ]
    return sorted(pairs, key=_canonical)


def _object_commitment(value: Any, seen: frozenset[int], global_depth: int) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _commitment_value(asdict(value), seen, global_depth)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _commitment_value(tolist(), seen, global_depth)
    state = _object_state(value, seen, global_depth)
    if state is not None:
        return {"type": _logical_name(value), "state": state}
    return {"opaque_type": _logical_name(value)}


def _code_hash(code: CodeType | None) -> str | None:
    if not isinstance(code, CodeType):
        return None
    return hashlib.sha256(marshal.dumps(_stable_code(code))).hexdigest()


def _stable_code(code: CodeType) -> CodeType:
    constants = tuple(
        _stable_code(value) if isinstance(value, CodeType) else value
        for value in code.co_consts
    )
    return code.replace(co_filename="", co_consts=constants)


def _logical_name(value: Any) -> str:
    return (
        f"{getattr(value, '__module__', type(value).__module__)}."
        f"{getattr(value, '__qualname__', type(value).__qualname__)}"
    )


def _source_hash(path: str | None) -> str | None:
    if path is None:
        return None
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _source_path(target: Any) -> str | None:
    try:
        return inspect.getsourcefile(target)
    except (OSError, TypeError):
        return None


def _representation_name(name: str | None, *, custom: bool, default: str) -> str:
    if name is None:
        if custom:
            raise ValueError("representation must name a custom projection")
        return default
    if not name.strip():
        raise ValueError("representation must not be empty")
    return name


def _value_hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _freeze(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return _freeze(asdict(value))
    return _freeze_object(value)


def _freeze_object(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_freeze(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_freeze(item) for item in value), key=_canonical)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _freeze(tolist())
    if hasattr(value, "__dict__"):
        return _freeze(vars(value))
    return repr(value)


def _freeze_mapping(value: Mapping[Any, Any]) -> Any:
    if all(isinstance(key, str) for key in value):
        return {key: _freeze(value[key]) for key in sorted(value)}
    pairs = [[_freeze(key), _freeze(item)] for key, item in value.items()]
    return sorted(pairs, key=_canonical)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _failure(error: Exception, state: Any) -> dict[str, Any]:
    return _error(error) | {"state": _freeze(state)}


def _error(error: Exception) -> dict[str, str]:
    return {"error": type(error).__name__, "message": str(error)}


def _project(projection: Projection, state: State) -> Any:
    return _freeze(projection(copy.deepcopy(state)))


def _validate_action_cap(actions: tuple[Action, ...] | None, max_actions: int) -> None:
    if actions is not None and len(actions) > max_actions:
        raise ValueError(f"actions exceeds max_actions={max_actions}")


def _state_pair(states: Sequence[State]) -> tuple[State, State]:
    if len(states) != 2:
        raise ValueError("states must contain exactly two states")
    return states[0], states[1]


def _positive_bound(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be at least one")


def _identity(value: State) -> State:
    return value
