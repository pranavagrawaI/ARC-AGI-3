# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ephemeral NOOA CodeAct producer for one grounding artifact.

This is a copy-and-prune adaptation of the pinned NOOA ARC example and the
qualified 0002-NOOA control. Unlike 0002, no sandbox, Python name, or writable
workspace persists between artifacts.
"""

from __future__ import annotations

import os as _dependency_os
from typing import Annotated, Any, Literal

_dependency_os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
_dependency_os.environ["LITELLM_MODE"] = "PRODUCTION"
del _dependency_os

from nooa import hidden

with hidden:
    import ast
    import json
    import sysconfig
    from importlib import metadata as importlib_metadata
    from importlib.util import find_spec
    from pathlib import Path

    from nooa import Agent, EventQuery
    from nooa.config import CodeActConfig
    from nooa.config.truncation_config import CaptureConfig, TruncationConfig
    from nooa.runtime.method_wrapper import create_agent_method_wrapper
    from nooa.runtime.restrictions import DEFAULT_BLOCKED_MODULES, RestrictionsConfig
    from nooa.runtime.sandbox.config import FileRule, SandboxConfig
    from nooa.strategies import CodeActStrategy
    from nooa.strategies.codeact import _ReturnResultSignal
    from nooa.unifiedllm import Tool
    from pydantic import BaseModel, ConfigDict, Field, model_validator

    _BLOCKED_RUNTIME_MODULES = frozenset(
        {
            "ctypes", "_thread", "concurrent", "gc", "importlib", "inspect",
            "litellm", "multiprocessing", "openai", "os", "site", "subprocess",
            "sys", "threading",
        }
    )
    _DECLARED_DISTRIBUTION_ROOTS = frozenset({"PIL", "numpy"})
    _UNDECLARED_DISTRIBUTION_ROOTS = (
        frozenset(importlib_metadata.packages_distributions()) - _DECLARED_DISTRIBUTION_ROOTS
    )
    _GROUNDING_RESTRICTIONS = RestrictionsConfig(
        blocked_modules=DEFAULT_BLOCKED_MODULES | _BLOCKED_RUNTIME_MODULES,
        restricted_imports=(
            DEFAULT_BLOCKED_MODULES
            | _BLOCKED_RUNTIME_MODULES
            | _UNDECLARED_DISTRIBUTION_ROOTS
            | {"agent", "nooa"}
        ),
    )

    def _sandbox_read_allowlist() -> tuple[FileRule, ...]:
        paths = {Path(sysconfig.get_path("stdlib")), Path("/lib"), Path("/lib64")}
        for import_root in _DECLARED_DISTRIBUTION_ROOTS:
            spec = find_spec(import_root)
            if spec is None or spec.origin is None:
                raise RuntimeError(f"declared distribution unavailable: {import_root}")
            paths.add(Path(spec.origin).parent)
        numpy_spec = find_spec("numpy")
        if numpy_spec is not None and numpy_spec.origin is not None:
            numpy_libs = Path(numpy_spec.origin).parent.parent / "numpy.libs"
            if numpy_libs.exists():
                paths.add(numpy_libs)
        return tuple(FileRule(path=str(path), access="read") for path in sorted(paths) if path.exists())

    _SANDBOX_READS = _sandbox_read_allowlist()
    class _PythonCodeRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")
        code: str = Field(description="Python source to execute in this artifact's fresh sandbox.")

    class _ArtifactSandboxStrategy(CodeActStrategy):
        """Prune framework bindings and seed only empirical inspection functions."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._seeded_worker_pid: int | None = None

        async def _run_prefill(self, runtime: Any, call: Any, builtins: dict[str, Any], session: Any) -> None:
            await super()._run_prefill(runtime, call, builtins, session)
            executor = session.sandbox_executor
            process = executor._proc
            pid = process.pid if process is not None and process.is_alive() else None
            if type(pid) is int and pid == self._seeded_worker_pid:
                return
            artifact_input = call.session_locals.get("artifact_input")
            if not isinstance(artifact_input, dict):
                raise TypeError("artifact_input missing from generated-code call")
            prune = await executor.run_cell(
                "for _name in tuple(globals()):\n"
                "    if not _name.startswith('__') and _name != 'return_result':\n"
                "        globals().pop(_name, None)\n"
                "del _name",
                execution_count=0,
            )
            if prune.error is not None:
                raise RuntimeError(f"failed to prune sandbox namespace: {prune.error}")
            evidence = artifact_input.get("evidence", {})
            phase = artifact_input.get("phase")
            visible_frames = tuple(evidence.get("frames", ()))
            transitions = tuple(evidence.get("transitions", ()))
            probe_results = tuple(evidence.get("probe_results", ()))
            observation_id = artifact_input.get("current_observation_id")
            current_probe = next(
                (
                    item
                    for item in reversed(probe_results)
                    if item.get("after_id") == observation_id
                ),
                None,
            )
            accepted_claim = artifact_input.get("accepted_claim")
            prior_probe_result_id = (
                current_probe.get("evidence_id") if isinstance(current_probe, dict) else None
            )
            parent_claim_id = (
                accepted_claim.get("claim_id") if isinstance(accepted_claim, dict) else None
            )
            primary_region = (
                accepted_claim.get("primary_region") if isinstance(accepted_claim, dict) else None
            )
            evidence_state = {
                "current_frame": visible_frames[-1] if visible_frames else None,
                "visible_frame_window": visible_frames,
                "frame_summaries": tuple(evidence.get("frame_summaries", ())),
                "transitions": transitions,
                "current_transition": next(
                    (
                        item
                        for item in reversed(transitions)
                        if item.get("after_id") == observation_id
                    ),
                    None,
                ),
                "probe_results": probe_results,
                "current_probe_result": current_probe,
                "attempts": tuple(evidence.get("attempts", ())),
                "events": evidence.get("events", {}),
                "accepted_claim": accepted_claim,
                "legal_actions": tuple(artifact_input.get("legal_actions", ())),
            }
            phase_helpers = (
                "def make_mask(row, col, rows):\n"
                "    if not rows or not all(isinstance(value, str) for value in rows):\n"
                "        raise ValueError('rows must be nonempty strings')\n"
                "    width = len(rows[0])\n"
                "    if width < 1 or any(len(value) != width for value in rows):\n"
                "        raise ValueError('mask rows must have equal positive width')\n"
                "    if any(set(value) - {'#', '.'} for value in rows) or not any('#' in value for value in rows):\n"
                "        raise ValueError('mask rows require #/. and at least one #')\n"
                "    return {'bbox': {'row': row, 'col': col, 'height': len(rows), 'width': width}, 'rows': list(rows)}\n"
                "def account_prior(*, disposition, rationale, _prior_probe_result_id="
                + repr(prior_probe_result_id)
                + "):\n"
                "    if _prior_probe_result_id is None:\n"
                "        raise ValueError('no prior probe result requires accounting')\n"
                "    if disposition not in {'retain', 'revise', 'abandon'}:\n"
                "        raise ValueError('disposition must be retain, revise, or abandon')\n"
                "    if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 512:\n"
                "        raise ValueError('rationale must contain 1..512 characters')\n"
                "    return {'prior_probe_result_id': _prior_probe_result_id, "
                "'disposition': disposition, 'rationale': rationale.strip()}\n"
                "def provisional_heuristic(*, name, belief, scope, decision_role, reconsider_when):\n"
                "    letters = 'abcdefghijklmnopqrstuvwxyz'\n"
                "    if not isinstance(name, str) or not name or len(name) > 48 or name[0] not in letters "
                "or any(char not in letters + '0123456789_' for char in name):\n"
                "        raise ValueError('name must be a stable lowercase identifier')\n"
                "    limits = {'belief': (belief, 512), 'scope': (scope, 256), "
                "'reconsider_when': (reconsider_when, 256)}\n"
                "    if any(not isinstance(value, str) or not value.strip() or len(value) > limit "
                "for value, limit in limits.values()):\n"
                "        raise ValueError('heuristic text fields must be nonempty and bounded')\n"
                "    if decision_role not in {'active', 'watch', 'defer'}:\n"
                "        raise ValueError('decision_role must be active, watch, or defer')\n"
                "    return {'name': name, 'belief': belief.strip(), 'scope': scope.strip(), "
                "'decision_role': decision_role, 'reconsider_when': reconsider_when.strip()}\n"
                "def submit_grounding_claim(primary_region, alternatives, hypothesis_update=None, "
                "heuristic_updates=(), additional_evidence_refs=(), _observation_id="
                + repr(observation_id)
                + ", _parent_claim_id="
                + repr(parent_claim_id)
                + ", _prior_probe_result_id="
                + repr(prior_probe_result_id)
                + "):\n"
                "    if (_prior_probe_result_id is not None) != (hypothesis_update is not None):\n"
                "        raise ValueError('hypothesis_update is required exactly when a prior probe exists')\n"
                "    return {'observation_id': _observation_id, "
                "'primary_region': primary_region, 'alternatives': list(alternatives), "
                "'evidence_refs': [_observation_id, *list(additional_evidence_refs)], "
                "'parent_claim_id': _parent_claim_id, "
                "'hypothesis_update': hypothesis_update, 'heuristic_updates': list(heuristic_updates)}\n"
                if phase == "claim"
                else
                "def cell_after(*, d_row, d_col, value):\n"
                "    if type(d_row) is not int or type(d_col) is not int or type(value) is not int:\n"
                "        raise TypeError('d_row, d_col, and value must be integers')\n"
                "    if not 0 <= value <= 15:\n"
                "        raise ValueError('value must be between 0 and 15')\n"
                "    return {'d_row': d_row, 'd_col': d_col, 'value': value}\n"
                "def predict_cell_values(*assertions, _primary="
                + repr(primary_region)
                + "):\n"
                "    if len(assertions) == 1 and isinstance(assertions[0], list):\n"
                "        raise TypeError('pass cell_after(...) arguments directly, not a list')\n"
                "    if not 1 <= len(assertions) <= 16:\n"
                "        raise ValueError('predict_cell_values requires 1..16 cell_after(...) arguments')\n"
                "    primary = _primary\n"
                "    bbox = primary['bbox']\n"
                "    selected = {(row, col) for row, values in enumerate(primary['rows']) "
                "for col, marker in enumerate(values) if marker == '#'}\n"
                "    expanded = []\n"
                "    seen = set()\n"
                "    for assertion in assertions:\n"
                "        if not isinstance(assertion, dict) or set(assertion) != {'d_row', 'd_col', 'value'}:\n"
                "            raise TypeError('predict_cell_values accepts only cell_after(...) arguments')\n"
                "        d_row = assertion['d_row']\n"
                "        d_col = assertion['d_col']\n"
                "        value = assertion['value']\n"
                "        if (d_row, d_col) not in selected:\n"
                "            raise ValueError('cell_after offsets must select a # cell in accepted primary_region')\n"
                "        coordinate = (bbox['row'] + d_row, bbox['col'] + d_col)\n"
                "        if coordinate in seen:\n"
                "            raise ValueError('cell_after offsets must be distinct')\n"
                "        seen.add(coordinate)\n"
                "        expanded.append({'row': coordinate[0], 'col': coordinate[1], 'value': value})\n"
                "    expanded.sort(key=lambda item: (item['row'], item['col']))\n"
                "    min_row = min(item['row'] for item in expanded)\n"
                "    max_row = max(item['row'] for item in expanded)\n"
                "    min_col = min(item['col'] for item in expanded)\n"
                "    max_col = max(item['col'] for item in expanded)\n"
                "    height = max_row - min_row + 1\n"
                "    width = max_col - min_col + 1\n"
                "    if height * width > 64:\n"
                "        raise ValueError('predicted cell bounding box exceeds 64 cells')\n"
                "    coordinates = {(item['row'], item['col']) for item in expanded}\n"
                "    rows = [''.join('#' if (row, col) in coordinates else '.' "
                "for col in range(min_col, max_col + 1)) for row in range(min_row, max_row + 1)]\n"
                "    expected_after = {'bbox': {'row': min_row, 'col': min_col, "
                "'height': height, 'width': width}, 'rows': rows}\n"
                "    return {'kind': 'replace', 'source': 'primary_region', "
                "'expected_after': expected_after, 'assertions': expanded}\n"
                "def predict_translate(d_row, d_col):\n"
                "    return {'kind': 'translate', 'source': 'primary_region', "
                "'d_row': d_row, 'd_col': d_col}\n"
                "def predict_unchanged():\n"
                "    return {'kind': 'unchanged', 'source': 'primary_region'}\n"
                "def action_prediction(action, predicted_effect, data=None):\n"
                "    return {'action': action, 'data': data, "
                "'predicted_effect': predicted_effect}\n"
                "def commit_contrastive(action, predicted_effect, alternative_predictions, data=None, _claim_id="
                + repr(parent_claim_id)
                + "):\n"
                "    return {'claim_id': _claim_id, "
                "'action': action, 'data': data, 'predicted_effect': predicted_effect, "
                "'alternative_predictions': list(alternative_predictions)}\n"
            )
            seed = (
                "def inspect_evidence(_value=" + repr(evidence_state) + "):\n"
                "    return _value\n"
                + phase_helpers
            )
            seeded = await executor.run_cell(seed, execution_count=0)
            if seeded.error is not None:
                raise RuntimeError(f"failed to seed empirical interface: {seeded.error}")
            process = executor._proc
            pid = process.pid if process is not None else None
            if type(pid) is not int:
                raise RuntimeError("sandbox worker has no process identity after seeding")
            self._seeded_worker_pid = pid

        def _build_execute_python_tool(self) -> Tool:
            tool = super()._build_execute_python_tool()
            tool.description = (
                "Inspect public evidence and submit this phase's semantic decision by executing "
                "Python 3 in a fresh isolated workspace. inspect_evidence() returns one coherent "
                "snapshot with the current frame, visible history, current transition/probe, accepted "
                "claim, and legal actions. The host also supplies NumPy/Pillow and only the "
                "phase-appropriate atomic submission constructors. The host owns identifiers and "
                "schema bookkeeping. Calling submit_grounding_claim(...) or commit_contrastive(...) "
                "atomically ends the artifact with that submission. "
                "Every call consumes one bounded artifact iteration. Evidence remains readable so "
                "a validation error can be diagnosed and repaired. Nothing persists after this artifact."
            )
            tool.parameters_model = _PythonCodeRequest
            return tool


class BoundingBoxModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    row: int = Field(ge=0)
    col: int = Field(ge=0)
    height: int = Field(ge=1)
    width: int = Field(ge=1)


class CellMaskModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    bbox: BoundingBoxModel
    rows: list[
        Annotated[
            str,
            Field(
                pattern=r"^[#.]+$",
                description="One mask row: # selects a cell and . leaves it unselected.",
            ),
        ]
    ] = Field(
        min_length=1,
        description=(
            "Exactly bbox.height strings, each exactly bbox.width characters; "
            "use only # and ., with at least one # across the mask."
        ),
    )

    @model_validator(mode="after")
    def validate_dimensions_and_selection(self) -> CellMaskModel:
        if len(self.rows) != self.bbox.height or any(
            len(row) != self.bbox.width for row in self.rows
        ):
            raise ValueError("rows must exactly match bbox height and width")
        if not any("#" in row for row in self.rows):
            raise ValueError("mask must select at least one cell with #")
        return self


class HypothesisUpdateModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    prior_probe_result_id: str = Field(min_length=1)
    disposition: Literal["retain", "revise", "abandon"]
    rationale: str = Field(min_length=1, max_length=512)


class HeuristicUpdateModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,47}$")
    belief: str = Field(min_length=1, max_length=512)
    scope: str = Field(min_length=1, max_length=256)
    decision_role: Literal["active", "watch", "defer"]
    reconsider_when: str = Field(min_length=1, max_length=256)


class GroundingClaimDraftModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    observation_id: str = Field(min_length=1)
    primary_region: CellMaskModel
    alternatives: list[CellMaskModel] = Field(
        max_length=3,
        description=(
            "Zero to three genuine alternative candidate masks. Use [] when there is no "
            "distinct alternative. Every supplied alternative must differ from the primary "
            "and from every other alternative."
        ),
    )
    evidence_refs: list[Annotated[str, Field(min_length=1)]] = Field(
        min_length=1,
        description="Distinct identifiers of available evidence records used for this claim.",
    )
    parent_claim_id: str | None
    hypothesis_update: HypothesisUpdateModel | None = None
    heuristic_updates: list[HeuristicUpdateModel] = Field(
        default_factory=list, max_length=12
    )

    @model_validator(mode="after")
    def validate_distinct_claim_members(self) -> GroundingClaimDraftModel:
        masks = [self.primary_region, *self.alternatives]
        serialized = [json.dumps(mask.model_dump(), sort_keys=True) for mask in masks]
        if len(set(serialized)) != len(serialized):
            raise ValueError("primary and alternative masks must all be distinct")
        if len(set(self.evidence_refs)) != len(self.evidence_refs):
            raise ValueError("evidence_refs must be distinct")
        names = [item.name for item in self.heuristic_updates]
        if len(set(names)) != len(names):
            raise ValueError("heuristic update names must be distinct")
        return self


class TranslateEffectModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    kind: Literal["translate"]
    source: Literal["primary_region"]
    d_row: int
    d_col: int

    @model_validator(mode="after")
    def validate_nonzero(self) -> TranslateEffectModel:
        if self.d_row == 0 and self.d_col == 0:
            raise ValueError("translation displacement must be non-zero")
        return self


class UnchangedEffectModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    kind: Literal["unchanged"]
    source: Literal["primary_region"]


class CellAssertionModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    row: int = Field(ge=0)
    col: int = Field(ge=0)
    value: int = Field(ge=0, le=15)


class ReplaceEffectModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    kind: Literal["replace"]
    source: Literal["primary_region"]
    expected_after: CellMaskModel
    assertions: list[CellAssertionModel] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def validate_bounded_assertions(self) -> ReplaceEffectModel:
        if self.expected_after.bbox.height * self.expected_after.bbox.width > 64:
            raise ValueError("replace expected_after bbox exceeds 64 cells")
        coordinates = [(item.row, item.col) for item in self.assertions]
        if len(set(coordinates)) != len(coordinates):
            raise ValueError("replace assertions must have distinct coordinates")
        selected = {
            (self.expected_after.bbox.row + row, self.expected_after.bbox.col + col)
            for row, values in enumerate(self.expected_after.rows)
            for col, value in enumerate(values)
            if value == "#"
        }
        if any(coordinate not in selected for coordinate in coordinates):
            raise ValueError("replace assertions must lie inside selected expected_after cells")
        return self


PredictedEffectModel = Annotated[
    TranslateEffectModel | UnchangedEffectModel | ReplaceEffectModel,
    Field(discriminator="kind"),
]


def _effect_signature(effect: Any) -> tuple[Any, ...]:
    if isinstance(effect, TranslateEffectModel):
        return (effect.kind, effect.d_row, effect.d_col)
    if isinstance(effect, UnchangedEffectModel):
        return (effect.kind,)
    if isinstance(effect, ReplaceEffectModel):
        return (
            effect.kind,
            effect.expected_after.bbox.height,
            effect.expected_after.bbox.width,
            tuple(effect.expected_after.rows),
            tuple(sorted(item.value for item in effect.assertions)),
        )
    raise TypeError(f"unsupported effect model: {type(effect).__name__}")


class ActionPredictionDraftModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    action: int
    data: dict[str, Any] | None
    predicted_effect: PredictedEffectModel


class ProbeCommitmentDraftModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    claim_id: str
    action: int
    data: dict[str, Any] | None
    predicted_effect: PredictedEffectModel
    alternative_predictions: list[ActionPredictionDraftModel] = Field(
        min_length=1,
        max_length=3,
        description=(
            "Predictions for one to three distinct currently legal alternative actions on "
            "the same accepted primary region. At least one must differ from the selected effect."
        ),
    )

    @model_validator(mode="after")
    def validate_contrast(self) -> ProbeCommitmentDraftModel:
        actions = [self.action, *(item.action for item in self.alternative_predictions)]
        if len(set(actions)) != len(actions):
            raise ValueError("selected and alternative action identifiers must be distinct")
        selected = _effect_signature(self.predicted_effect)
        alternatives = [
            _effect_signature(item.predicted_effect) for item in self.alternative_predictions
        ]
        if all(effect == selected for effect in alternatives):
            raise ValueError("at least one alternative action must predict a different effect")
        return self


class MissingProviderUsage(RuntimeError):
    """A live-capable call omitted mechanical input/output token counters."""


class GroundingAgent(Agent):
    """NOOA method shell that produces one artifact and owns no action authority."""

    _artifact_input: Annotated[dict[str, Any], hidden]
    _produce_claim: Annotated[Any, hidden]
    _produce_commitment: Annotated[Any, hidden]
    _require_usage: Annotated[bool, hidden]
    _usage: Annotated[list[dict[str, Any]], hidden]
    _artifact_serial: Annotated[int, hidden]

    def __init__(
        self,
        *args: Any,
        workspace: str,
        require_usage: bool = False,
        max_iterations: int = 10,
        max_retries: int = 1,
        timeout_seconds: float = 20.0,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("event_query", EventQuery.current_call())
        kwargs.setdefault("truncation", TruncationConfig(capture=CaptureConfig(max_stdout=4_000)))
        super().__init__(*args, **kwargs)
        self._artifact_input = {}
        self._require_usage = require_usage
        self._usage = []
        self._artifact_serial = 0

        def make_strategy() -> _ArtifactSandboxStrategy:
            return _ArtifactSandboxStrategy(
                CodeActConfig(
                    max_iterations=max_iterations,
                    max_retries=max_retries,
                    cell_timeout=timeout_seconds,
                    prefill=None,
                    execution_backend="sandbox",
                    sandbox=SandboxConfig(
                        workspace=workspace,
                        allow=_SANDBOX_READS,
                        system_paths=False,
                        network=False,
                        max_memory_mb=512,
                        max_cpu_seconds=max(1, int(timeout_seconds)),
                        recovery="restart_empty",
                        require=True,
                        context_block=False,
                    ),
                    restrictions=_GROUNDING_RESTRICTIONS,
                )
            )

        claim_strategy = make_strategy()
        commitment_strategy = make_strategy()

        async def analyze(self: Any, artifact_input: dict[str, Any]) -> GroundingClaimDraftModel:
            """Inspect immutable public evidence and return one candidate claim without an action."""

        async def commit(self: Any, artifact_input: dict[str, Any]) -> ProbeCommitmentDraftModel:
            """Bind the accepted claim to one legal action and one exact prospective effect."""

        self._produce_claim = create_agent_method_wrapper(
            analyze, needs_generation=True, needs_tracing=True, strategy=claim_strategy
        ).__get__(self, type(self))
        self._produce_commitment = create_agent_method_wrapper(
            commit, needs_generation=True, needs_tracing=True, strategy=commitment_strategy
        ).__get__(self, type(self))

        async def accept_atomic_submission(context: Any, next_handler: Any) -> Any:
            """Make a submit/commit constructor call terminal in its containing statement."""
            try:
                tree = ast.parse(context.code)
            except (SyntaxError, TypeError):
                return await next_handler(context)

            def submission_call(value: Any) -> Any | None:
                if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name):
                    return None
                if value.func.id in {"submit_grounding_claim", "commit_contrastive"}:
                    return value
                if (
                    value.func.id == "return_result"
                    and len(value.args) == 1
                    and not value.keywords
                ):
                    return submission_call(value.args[0])
                return None

            submission_index: int | None = None
            submission: Any | None = None
            target_name: str | None = None
            for index, statement in enumerate(tree.body):
                if isinstance(statement, ast.Expr):
                    candidate = submission_call(statement.value)
                elif (
                    isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                ):
                    candidate = submission_call(statement.value)
                    target_name = statement.targets[0].id if candidate is not None else None
                else:
                    candidate = None
                if candidate is not None:
                    submission_index = index
                    submission = candidate
                    break
            if submission_index is None or submission is None:
                return await next_handler(context)

            result_name = target_name or "_grounding_artifact_result"
            tree.body[submission_index:] = [
                ast.Assign(
                    targets=[ast.Name(id=result_name, ctx=ast.Store())],
                    value=submission,
                ),
                ast.Expr(value=ast.Name(id=result_name, ctx=ast.Load())),
            ]
            ast.fix_missing_locations(tree)
            context.code = ast.unparse(tree)
            processed = await next_handler(context)
            execution = processed.result
            if execution.error is None:
                execution.signal = _ReturnResultSignal(result=execution.returned_value)
            return processed

        self.event_manager.intercept("execute_python", accept_atomic_submission)

        async def freeze_model_surface(context: Any, next_handler: Any) -> Any:
            artifact_input = self._artifact_input
            phase = artifact_input.get("phase")
            contract = artifact_input.get("model_contract")
            run_policy = artifact_input.get("run_policy")
            if phase not in {"claim", "commitment"} or not isinstance(contract, dict):
                raise RuntimeError("invalid artifact input before model call")
            if not isinstance(run_policy, dict):
                raise TypeError("run policy missing before model call")
            long_run = run_policy.get("mode") == "autonomous_44_action"
            continuation = (
                "In this frozen long-run condition, an ordinary contradicted or exhausted "
                "grounding cycle is recorded and reset without human input. G2 is recorded but "
                "opens no new API; the same grounding-only cycle begins again. Use the probe_results "
                "inside inspect_evidence() "
                "to revise your next hypothesis from prior commitments and verdicts. Provider, "
                "artifact, environment, death, success, budget, and action-cap stops remain terminal."
                if long_run
                else
                "In this review-stop condition, a consequential failure closes the scored run "
                "rather than letting a human repair it."
            )
            common = """
You are the semantic investigator inside an evidence-preserving action harness.

WHY THE HARNESS EXISTS
The experiment asks whether your visual hypothesis survives an action chosen in advance. The
host therefore separates immutable public evidence, your hypothesis, your prospective prediction,
the single environment action, and the later verdict. This prevents hindsight from rewriting what
you predicted. RUN_CONTINUATION_POLICY
The verifier is intentionally non-oracular: it can check your claim against public successor
evidence, but it never tells you the correct object, action, route, or world semantics.

DIVISION OF WORK
Do the semantic work deterministic code cannot do: inspect the grid and history, choose a plausible
controllable region, distinguish real alternatives, choose an informative legal action, and predict
the claimed region's exact action-conditioned change. The host does mechanical work: immutable
record IDs, current-observation citation, claim_id binding, schema fields, legality checks, one-action
authority, evidence recording, and verdict calculation. Do not manually reproduce those mechanics.

AVAILABLE EVIDENCE
inspect_evidence() returns one snapshot with current_frame, visible_frame_window, frame_summaries,
transitions, current_transition, probe_results, current_probe_result, attempts, events,
accepted_claim, and legal_actions. The two current_* history fields are already matched to the
current frame and are null when absent. A probe result links its prior claim, commitment, and
non-oracular verdict. A frame contains a zero-indexed integer grid and public metadata.
Action integers are opaque: their
identifiers alone do not imply movement or any other effect. NumPy and Pillow may be imported for
analysis. The sandbox is fresh for every claim or commitment; preserve nothing in it.

DECISION DISCIPLINE
Do not print or restate a whole grid: it is already visible through inspect_evidence(). On later cycles,
read the latest transition diff and probe result first. Every execute_python call consumes one of the
artifact's bounded iterations, but evidence remains readable after an error so you can diagnose and
repair the submission. Exploration should compute a discriminating fact, not narrate
candidate lists. When several hypotheses remain, choose the best falsifiable one and submit it; the
point of a prospective probe is to resolve uncertainty through action rather than prose.

PROVISIONAL KNOWLEDGE
Conserve observations while treating every interpretation as a scoped, revisable heuristic. Visible
change outside a predicted region may be independent environment dynamics and need not be absorbed
into the chosen-action model. It also must not disappear from the evidence. A heuristic may guide,
watch, or defer attention; none may be called explained, closed, confirmed, or certain. Record what
you currently believe, the observations and scope supporting it, and what would make you reconsider.
The host preserves observations and checks only temporal/evidence binding; you decide grouping,
relevance, identity, causal role, and which provisional heuristic should guide the next action.

WORKFLOW
1. Claim phase: identify the best candidate region from public evidence; no action is chosen yet.
2. The host validates it and assigns claim_id.
3. Commitment phase: bind that accepted region to one legal action and one prospective effect.
4. Only then may the host execute exactly one ARC action and compare the successor with the prediction.

Only execute_python is callable. Complete the artifact inside that tool call. The functions named
submit_grounding_claim() and commit_contrastive() submit immediately and end the artifact; do not
wrap them in return_result() or expect to continue to another phase in the same artifact. All named
evidence and constructor functions are already bound Python globals;
call them directly and do not import them as modules. Constructors intentionally make host-owned
fields for you. This artifact permits exactly ITERATION_BUDGET execute_python calls. Reserve a call
for submission; exploration that consumes the final call produces no artifact.
""".replace("RUN_CONTINUATION_POLICY", continuation).replace(
                "ITERATION_BUDGET", str(max_iterations)
            ).strip()
            phase_instructions = (
                """
CLAIM PHASE
Use make_mask(row, col, rows) where each row is a #/. string: # selects a cell and . leaves it out.
The bounding box dimensions are derived mechanically. Submit the strongest candidate and zero to
three genuinely distinct alternatives. The host inserts current observation and parent identifiers.
Only pass additional_evidence_refs when another returned evidence record was materially used.
After any prior probe, change_accounting reports only visible counts inside and outside the prior
primary mask. Outside change is neither a debt that must be explained nor evidence that may be erased.
Use account_prior() to author whether you retain, revise, or abandon the prior hypothesis. Maintain a
provisional interpretation with provisional_heuristic(), but submit only heuristics whose
interpretation changes. The host retains omitted heuristics and unions their prior evidence with the
current claim and probe; do not copy host-owned evidence identifiers or the full register.
Use active for a current decision premise, watch for a background regularity whose violation should
resurface, and defer when evidence is insufficient. Every heuristic must state reconsider_when.
Finish by calling submit_grounding_claim(...) in the same call that constructs the claim.

Example form (coordinates and shape are illustrative, not hints about this grid):
frame = inspect_evidence()['current_frame']
grid = frame['grid']
primary = make_mask(row=4, col=7, rows=['.#.', '###'])
alternative = make_mask(row=10, col=2, rows=['##'])
submit_grounding_claim(primary_region=primary, alternatives=[alternative])

After a prior probe, add an explicit update and only the changed heuristic. Words are illustrative:
update = account_prior(disposition='revise',
    rationale='The prior effect mismatched this transition; I am testing another grouping.')
candidate = provisional_heuristic(name='candidate_motion',
    belief='The selected pattern may move under the tested action.',
    scope='Current level; one observed transition.',
    decision_role='active', reconsider_when='The same prospective effect mismatches.')
submit_grounding_claim(primary_region=primary, alternatives=[alternative],
    hypothesis_update=update, heuristic_updates=[candidate])

Do not choose an action in this phase.
""".strip()
                if phase == "claim"
                else
                """
COMMITMENT PHASE
accepted_claim is available in inspect_evidence() and already has its host claim_id. Choose exactly one
integer from legal_actions for execution. Also predict the same accepted primary region under one
to three distinct currently legal alternative actions. At least one alternative must have a
different position-invariant effect from the selected action. The contrast is persisted so a later
fresh artifact can select the alternative and prospectively close the same two-action map.

An observed effect can be locally supported while action specificity remains inconclusive. A
prediction expected under every legal action cannot promote. Action specificity becomes supported
only after two different selected actions both succeed and each commitment had prospectively
predicted the other's effect. Use inspect_evidence()['probe_results'] to preserve and test that map. Do not discard a
coherent action-varying change merely because a smaller action-insensitive region is predictable.

Effect constructors anchor every prediction to accepted_claim.primary_region. The atomic
commit_contrastive helper inserts claim_id and binds the selected and alternative predictions.
Finish by calling commit_contrastive(...) in the same call that constructs the commitment.

Example form (numbers are illustrative, not action semantics):
selected = predict_cell_values(cell_after(d_row=1, d_col=1, value=3))
contrast = action_prediction(action=5, predicted_effect=predict_unchanged(), data=None)
commit_contrastive(action=2, predicted_effect=selected,
    alternative_predictions=[contrast], data=None)

cell_after() uses offsets from the top-left of accepted_claim.primary_region, not absolute frame
coordinates. Each offset must name a # cell in that primary mask. predict_cell_values() predicts
values at those same fixed coordinates in the successor frame; it does not track an object or its
destination. Pass one to sixteen cell_after(...) arguments directly, with no list wrapper. Use
predict_translate() when the same selected pattern is expected to move as a unit.
predict_unchanged() is useful for falsification but earns no promotion by itself. Prefer informative
anchored effects when evidence supports them. Never predict an unrelated HUD change.
""".strip()
            )
            system = (
                common
                + "\n\nPUBLIC CONTRACT\n"
                + json.dumps(contract, sort_keys=True, separators=(",", ":"))
                + "\n\nARTIFACT PHASE: "
                + phase
                + "\n"
                + phase_instructions
            )
            task = {
                "role": "user",
                "content": json.dumps(
                    {
                        "phase": phase,
                        "current_observation_id": artifact_input.get("current_observation_id"),
                        "legal_actions": artifact_input.get("legal_actions", []),
                        "accepted_claim": artifact_input.get("accepted_claim"),
                        "run_policy": run_policy,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
            repaired_context = [{"role": "system", "content": system}, task]
            for message in context.messages:
                content = message.get("content")
                if message.get("role") == "system":
                    continue
                if isinstance(content, str) and "Task(prompt=" in content:
                    continue
                if isinstance(content, str) and content.lstrip().startswith("<context>"):
                    continue
                repaired_context.append(message)
            context.messages = repaired_context
            processed = await next_handler(context)
            usage = getattr(processed.response, "usage", None)
            if self._require_usage:
                if not isinstance(usage, dict):
                    raise MissingProviderUsage("provider response has no usage object")
                input_tokens = usage.get("input_tokens")
                output_tokens = usage.get("output_tokens")
                if type(input_tokens) is not int or input_tokens < 0 or type(output_tokens) is not int or output_tokens < 0:
                    raise MissingProviderUsage("provider must report non-negative input_tokens and output_tokens")
                output_details = usage.get("output_tokens_details")
                input_details = usage.get("input_tokens_details")
                self._usage.append(
                    {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "reasoning_tokens": (
                            usage.get("reasoning_tokens")
                            if usage.get("reasoning_tokens") is not None
                            else output_details.get("reasoning_tokens")
                            if isinstance(output_details, dict)
                            else None
                        ),
                        "cache_read_tokens": (
                            usage.get("cache_read_tokens")
                            if usage.get("cache_read_tokens") is not None
                            else input_details.get("cached_tokens")
                            if isinstance(input_details, dict)
                            else None
                        ),
                        "cache_write_tokens": usage.get("cache_write_tokens"),
                    }
                )
            return processed

        self.event_manager.intercept("llm_call", freeze_model_surface)

    async def produce(self, artifact_input: dict[str, Any]) -> dict[str, Any]:
        self._artifact_input = artifact_input
        self._artifact_serial += 1
        phase = artifact_input.get("phase")
        if phase == "claim":
            result = await self._produce_claim(artifact_input, _session_locals={"artifact_input": artifact_input})
        elif phase == "commitment":
            result = await self._produce_commitment(artifact_input, _session_locals={"artifact_input": artifact_input})
        else:
            raise ValueError("phase must be claim or commitment")
        return result.model_dump(mode="json")

    @property
    def usage(self) -> list[dict[str, Any]]:
        return list(self._usage)
