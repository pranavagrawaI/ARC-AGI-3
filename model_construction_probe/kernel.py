"""Offline prediction-boundary prototype. Python 3.11+, standard library only.

This module is NOT a sandbox for hostile generated Python. It deliberately accepts
only self-contained source, evaluates calls in fresh namespaces, and snapshots I/O
as JSON. Production must execute these calls inside its existing isolated realm.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping


def pack(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(pack(value).encode()).hexdigest()


class ContractError(ValueError):
    pass


class ModelMiss(RuntimeError):
    """No maintained candidate explains the observation. Not environment failure."""


class BranchLimit(RuntimeError):
    """An explicit compute limit, never evidence for eliminating a candidate."""


@dataclass(frozen=True)
class Observation:
    raw_json: str
    measured_json: str

    @classmethod
    def make(cls, raw: Any, measured: Mapping[str, Any]) -> Observation:
        # Measured targets belong to a fixed evaluator, not to the candidate model.
        return cls(pack(raw), pack(dict(measured)))


@dataclass(frozen=True)
class Particle:
    candidate: str
    lineage: tuple[int, ...]
    state_json: str


@dataclass(frozen=True)
class Belief:
    model_id: str
    frontier: int
    particles: tuple[Particle, ...]


@dataclass(frozen=True)
class Branch:
    particle: Particle
    claims_json: str
    unresolved: tuple[str, ...]


@dataclass(frozen=True)
class Forecast:
    model_id: str
    frontier: int
    prior_id: str
    action_json: str
    branches: tuple[Branch, ...]

    @property
    def seal(self) -> str:
        return digest(asdict(self))


@dataclass(frozen=True)
class Assessment:
    forecast_id: str
    observation_id: str
    status: str
    survivors: tuple[tuple[int, ...], ...]
    fields_json: str
    coverage: float
    specificity: float


@dataclass(frozen=True)
class Targets:
    """Predeclared finite observable domains, for diagnostic fixtures only.

    Missing measurements are not negative observations. Missing claims and claims
    spanning the full domain receive no information credit. This is a diagnostic
    metric, not a proper probabilistic score or a general ARC evaluation metric.
    """
    domains_json: str

    @classmethod
    def make(cls, domains: Mapping[str, Iterable[Any]]) -> Targets:
        result = {k: list(v) for k, v in domains.items()}
        if not result:
            raise ContractError("At least one evaluator-owned target is required")
        for key, values in result.items():
            if len(values) < 2 or len({pack(v) for v in values}) != len(values):
                raise ContractError(f"{key}: domain needs at least two unique values")
        return cls(pack(result))


@dataclass(frozen=True)
class FrozenModel:
    source: str

    def __post_init__(self) -> None:
        tree = ast.parse(self.source)
        if any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(tree)):
            raise ContractError("Prototype freezes one self-contained module; imports unsupported")
        for name in ("initialize", "predict_one", "observe_one"):
            if not any(isinstance(n, ast.FunctionDef) and n.name == name for n in tree.body):
                raise ContractError(f"Missing required function: {name}")

    @property
    def model_id(self) -> str:
        return hashlib.sha256(self.source.encode()).hexdigest()

    def call(self, name: str, *args: Any) -> Any:
        # Fresh namespace prevents ordinary module-global mutation from becoming
        # hidden persistent state. Restrictions here are NOT a security sandbox.
        namespace: dict[str, Any] = {}
        exec(compile(self.source, "<frozen-world-model>", "exec"), namespace)
        result = namespace[name](*json.loads(pack(args)))
        return json.loads(pack(result))


def initialize(model: FrozenModel, observation: Observation, cap: int = 32) -> Belief:
    roots = model.call("initialize", json.loads(observation.raw_json))
    if not isinstance(roots, list) or not roots:
        raise ContractError("initialize must return a nonempty list of candidates")
    if len(roots) > cap:
        raise BranchLimit("Initial candidate cap exceeded; none silently discarded")
    particles = tuple(
        Particle(str(r["candidate"]), (i,), pack(r["state"])) for i, r in enumerate(roots)
    )
    return Belief(model.model_id, 0, particles)


def predict(model: FrozenModel, belief: Belief, action: Any, targets: Targets,
            cap: int = 32) -> Forecast:
    if belief.model_id != model.model_id:
        raise ContractError("Model version changed: rebuild belief from the visible prefix")
    if not belief.particles:
        raise ModelMiss("Cannot roll out an empty belief")
    domains = json.loads(targets.domains_json)
    branches = []
    for parent in belief.particles:
        outcomes = model.call("predict_one", parent.candidate,
                              json.loads(parent.state_json), action)
        if not isinstance(outcomes, list) or not outcomes:
            raise ContractError("Return a branch with named unresolved dependencies, not []")
        for index, outcome in enumerate(outcomes):
            claims = outcome.get("claims", {})
            unresolved = tuple(outcome.get("unresolved", []))
            if not claims and not unresolved:
                raise ContractError("An uninformative branch must name its unresolved dependency")
            if any(not isinstance(u, str) or not u for u in unresolved):
                raise ContractError("Unresolved dependencies must have nonempty names")
            for key, allowed in claims.items():
                if key not in domains or not isinstance(allowed, list) or not allowed:
                    raise ContractError(f"Invalid claim: {key}")
                domain = {pack(v) for v in domains[key]}
                if not {pack(v) for v in allowed} <= domain:
                    raise ContractError(f"Claim outside evaluator-owned domain: {key}")
            # The host preserves candidate and ancestry; the model supplies only state.
            particle = Particle(parent.candidate, parent.lineage + (index,),
                                pack(outcome["state"]))
            branches.append(Branch(particle, pack(claims), unresolved))
            if len(branches) > cap:
                raise BranchLimit("Rollout cap exceeded; none silently discarded")
    return Forecast(model.model_id, belief.frontier + 1, digest(asdict(belief)),
                    pack(action), tuple(branches))


def assess(forecast: Forecast, observation: Observation, targets: Targets) -> Assessment:
    measured = json.loads(observation.measured_json)
    domains = json.loads(targets.domains_json)
    if not set(measured) <= set(domains):
        raise ContractError("Measured target absent from the frozen target contract")
    for key, value in measured.items():
        if pack(value) not in {pack(v) for v in domains[key]}:
            raise ContractError(f"Observed value outside domain: {key}")
    claims = [json.loads(branch.claims_json) for branch in forecast.branches]
    survivors = tuple(
        b.particle.lineage for b, c in zip(forecast.branches, claims)
        if all(k not in c or pack(v) in {pack(x) for x in c[k]}
               for k, v in measured.items())
    )
    fields = {}
    for key, truth in measured.items():
        domain = {pack(v) for v in domains[key]}
        allowed: set[str] = set()
        for c in claims:
            # An abstaining alternative cannot be hidden behind a narrower one.
            allowed |= {pack(v) for v in c[key]} if key in c else domain
        informative = 0 < len(allowed) < len(domain)
        specificity = math.log(len(domain) / len(allowed)) / math.log(len(domain)) if allowed else 0.0
        fields[key] = {
            "covered": informative, "hit": pack(truth) in allowed,
            "width": len(allowed), "domain_size": len(domain),
            "specificity": specificity,
        }
    denominator = len(fields) or 1
    coverage = sum(f["covered"] for f in fields.values()) / denominator
    specificity = sum(f["specificity"] for f in fields.values()) / denominator
    status = "contradicted" if not survivors else "consistent" if coverage else "unresolved"
    return Assessment(forecast.seal, digest(asdict(observation)), status,
                      survivors, pack(fields), coverage, specificity)


def assimilate(model: FrozenModel, forecast: Forecast, observation: Observation,
               assessment: Assessment) -> Belief:
    # Prediction is assessed BEFORE observe_one can inspect the successor.
    if model.model_id != forecast.model_id:
        raise ContractError("Model revision cannot inherit another version's forecast")
    if assessment.forecast_id != forecast.seal or assessment.observation_id != digest(asdict(observation)):
        raise ContractError("Assessment is bound to a different forecast or observation")
    if not assessment.survivors:
        raise ModelMiss("All maintained candidates contradicted; start a model-repair work item")
    particles = []
    for b in forecast.branches:
        p = b.particle
        if p.lineage in assessment.survivors:
            state = model.call("observe_one", p.candidate, json.loads(p.state_json),
                               json.loads(observation.raw_json))
            particles.append(Particle(p.candidate, p.lineage, pack(state)))
    return Belief(model.model_id, forecast.frontier, tuple(particles))


def advance_without_observation(forecast: Forecast) -> Belief:
    return Belief(forecast.model_id, forecast.frontier,
                  tuple(b.particle for b in forecast.branches))


@dataclass(frozen=True)
class Step:
    action_json: str
    observation: Observation

    @classmethod
    def make(cls, action: Any, observation: Observation) -> Step:
        return cls(pack(action), observation)


@dataclass(frozen=True)
class Prefix:
    initial: Observation
    steps: tuple[Step, ...]

    @property
    def frontier(self) -> int:
        return len(self.steps)

    def require_refs(self, refs: Iterable[int]) -> None:
        if any(type(r) is not int or not 0 <= r <= self.frontier for r in refs):
            raise ContractError("Evidence reference beyond visible prefix")


def rebuild(model: FrozenModel, prefix: Prefix, targets: Targets) -> Belief:
    """Reconstruct under this model version; known contradictions cannot be swallowed."""
    belief = initialize(model, prefix.initial)
    for step in prefix.steps:
        f = predict(model, belief, json.loads(step.action_json), targets)
        verdict = assess(f, step.observation, targets)
        belief = assimilate(model, f, step.observation, verdict)
    return belief


def score_one_step(model: FrozenModel, belief: Belief, steps: tuple[Step, ...],
                   targets: Targets) -> tuple[Assessment, ...]:
    results = []
    for step in steps:
        f = predict(model, belief, json.loads(step.action_json), targets)
        verdict = assess(f, step.observation, targets)
        results.append(verdict)
        if verdict.status == "contradicted":
            break
        belief = assimilate(model, f, step.observation, verdict)
    return tuple(results)


def score_open_loop(model: FrozenModel, belief: Belief, steps: tuple[Step, ...],
                    targets: Targets, cap: int = 32) -> tuple[Assessment, ...]:
    """Build the entire action-conditioned tree BEFORE reading any suffix observation.

    Audit then preserves path compatibility. It neither rewrites predicted state
    nor calls observe_one. A later match cannot resurrect an earlier invalid path.
    This is a fixed-action test, NOT an off-policy test of a new actor's decisions.
    """
    forecasts = []
    for step in steps:
        f = predict(model, belief, json.loads(step.action_json), targets, cap)
        forecasts.append(f)
        belief = advance_without_observation(f)
    results = []
    viable: set[tuple[int, ...]] | None = None
    for f, step in zip(forecasts, steps):
        verdict = assess(f, step.observation, targets)
        current = set(verdict.survivors)
        if viable is not None:
            current = {path for path in current if path[:-1] in viable}
        viable = current
        if not viable:
            verdict = Assessment(verdict.forecast_id, verdict.observation_id,
                                 "contradicted", (), verdict.fields_json,
                                 verdict.coverage, verdict.specificity)
        results.append(verdict)
        if not viable:
            break
    return tuple(results)
