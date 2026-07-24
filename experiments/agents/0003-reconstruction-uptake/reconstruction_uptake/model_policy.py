"""OpenAI-compatible replay policy with controller-mediated artifact execution."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Protocol, Sequence

from .runner import PolicyPrediction, PolicyUpdate


PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "python",
        "description": (
            "Execute Python 3 in the persistent isolated workspace. Files and Python state "
            "persist for this replay; the repository, game implementation, and network are absent."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python source to execute."}
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    },
}
_RESULT_MARKER = "__ARC_RECONSTRUCTION_RESULT__"


class _Response(Protocol):
    message: Mapping[str, Any]
    usage: Mapping[str, Any]


class _Client(Protocol):
    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> _Response: ...


class _WorkspaceResult(Protocol):
    ok: bool
    stdout: str
    stderr: str

    def model_text(self) -> str: ...


class _Workspace(Protocol):
    def execute(self, code: str) -> _WorkspaceResult: ...


class ModelReplayPolicy:
    """Let a model edit an artifact, then derive commitments by executing that artifact."""

    def __init__(
        self,
        *,
        client: _Client,
        workspace: _Workspace,
        condition: str,
        system_prompt: str,
        max_rounds: int = 5,
        max_context_turns: int = 6,
    ) -> None:
        if condition not in {"optional", "required"}:
            raise ValueError("condition must be 'optional' or 'required'")
        if max_rounds < 1 or max_context_turns < 1:
            raise ValueError("model round and context limits must be positive")
        self.client = client
        self.workspace = workspace
        self.condition = condition
        self.system_prompt = system_prompt
        self.max_rounds = max_rounds
        self.max_context_turns = max_context_turns
        self.model_calls = 0
        self.python_calls = 0
        self.provider_usage: list[dict[str, Any]] = []
        self.request_evidence: list[dict[str, Any]] = []
        self._turns: list[list[dict[str, Any]]] = []
        self._last_prediction: Mapping[str, Any] | None = None
        self._last_update: Mapping[str, Any] | None = None

    def predict(
        self,
        *,
        step: int,
        observation: dict[str, Any],
        recorded_action: int,
        legal_actions: tuple[int, ...],
    ) -> PolicyPrediction:
        self._write_observation(observation)
        turn: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": self._prediction_prompt(
                    step=step,
                    recorded_action=recorded_action,
                    legal_actions=legal_actions,
                ),
            }
        ]
        final: Mapping[str, Any] | None = None
        for model_round in range(1, self.max_rounds + 1):
            self.model_calls += 1
            try:
                response = self.client.complete(
                    messages=self._messages(turn),
                    tools=(PYTHON_TOOL,),
                )
            except Exception as error:
                self.request_evidence.append(
                    _transport_evidence(error, model_call=self.model_calls, usage={})
                )
                raise
            self.provider_usage.append(dict(response.usage))
            self.request_evidence.append(
                _transport_evidence(
                    response,
                    model_call=self.model_calls,
                    usage=dict(response.usage),
                )
            )
            assistant = dict(response.message)
            assistant.setdefault("role", "assistant")
            turn.append(assistant)
            tool_calls = assistant.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                if model_round == self.max_rounds:
                    raise RuntimeError("model exhausted replay deliberation rounds with tool calls")
                turn.extend(self._execute_tool_calls(tool_calls, model_round=model_round))
                continue
            final = assistant
            break
        if final is None:
            raise RuntimeError("model did not produce a final replay prediction")

        direct = _json_object(_message_text(final.get("content")))
        artifact = self._invoke_artifact(legal_actions)
        if artifact is not None and artifact.loaded:
            candidates = artifact.candidate_predictions
            prediction = candidates.get(str(recorded_action))
            if not isinstance(prediction, Mapping):
                raise RuntimeError("artifact did not return an object for the recorded action")
            result = PolicyPrediction(
                prediction=dict(prediction),
                candidate_predictions=candidates,
                artifact_sha256=artifact.sha256,
                artifact_entrypoints=artifact.entrypoints,
                artifact_loaded=True,
            )
        else:
            direct_prediction = direct.get("prediction")
            direct_candidates = direct.get("candidate_predictions", {})
            if not isinstance(direct_prediction, Mapping):
                raise RuntimeError("final model response requires an object prediction")
            if not isinstance(direct_candidates, Mapping):
                raise RuntimeError("candidate_predictions must be an object")
            result = PolicyPrediction(
                prediction=dict(direct_prediction),
                candidate_predictions={
                    str(action): dict(value)
                    for action, value in direct_candidates.items()
                    if isinstance(value, Mapping)
                },
                artifact_sha256=artifact.sha256 if artifact is not None else None,
                artifact_entrypoints=artifact.entrypoints if artifact is not None else (),
                artifact_loaded=False,
            )

        compact_turn = [turn[0], dict(final)]
        self._turns.append(compact_turn)
        self._turns = self._turns[-self.max_context_turns :]
        self._last_prediction = dict(result.prediction)
        return result

    def update(
        self,
        *,
        step: int,
        recorded_action: int,
        observation: dict[str, Any],
    ) -> PolicyUpdate:
        self._write_observation(observation)
        result = self.workspace.execute(_update_script(recorded_action))
        self.python_calls += 1
        payload = _marked_payload(result.stdout) if result.ok else None
        if not isinstance(payload, Mapping):
            detail = {
                "workspace_ok": result.ok,
                "stderr": result.stderr[:2000],
            }
            self._last_update = detail
            return PolicyUpdate(ok=False, detail=detail)
        detail = dict(payload)
        self._last_update = detail
        return PolicyUpdate(ok=payload.get("updated") is True, detail=detail)

    def _write_observation(self, observation: Mapping[str, Any]) -> None:
        encoded = json.dumps(observation, separators=(",", ":"), sort_keys=True)
        result = self.workspace.execute(
            "from pathlib import Path\n"
            f"Path('current_observation.json').write_text({encoded!r}, encoding='utf-8')"
        )
        self.python_calls += 1
        if not result.ok:
            raise RuntimeError(f"could not write replay observation: {result.stderr[:2000]}")

    def _invoke_artifact(self, legal_actions: tuple[int, ...]) -> "_ArtifactResult | None":
        result = self.workspace.execute(_artifact_script(legal_actions))
        self.python_calls += 1
        payload = _marked_payload(result.stdout) if result.ok else None
        if not isinstance(payload, Mapping) or payload.get("exists") is not True:
            return None
        digest = payload.get("artifact_sha256")
        entrypoints = payload.get("entrypoints")
        predictions = payload.get("candidate_predictions")
        if not isinstance(digest, str) or not isinstance(entrypoints, list):
            return None
        clean_predictions = (
            {
                str(action): dict(value)
                for action, value in predictions.items()
                if isinstance(value, Mapping)
            }
            if isinstance(predictions, Mapping)
            else {}
        )
        return _ArtifactResult(
            loaded=payload.get("loaded") is True,
            sha256=digest,
            entrypoints=tuple(str(name) for name in entrypoints),
            candidate_predictions=clean_predictions,
        )

    def _messages(self, turn: list[dict[str, Any]]) -> list[dict[str, Any]]:
        history = [message for prior in self._turns for message in prior]
        return [
            {"role": "system", "content": self.system_prompt},
            *history,
            *turn,
        ]

    def _prediction_prompt(
        self,
        *,
        step: int,
        recorded_action: int,
        legal_actions: tuple[int, ...],
    ) -> str:
        common = (
            f"Offline replay step {step}. The current admissible observation is in "
            "current_observation.json. The recorded action to predict is "
            f"{recorded_action}; legal actions are {list(legal_actions)}. "
            "The next observation is concealed until your prediction is committed. "
        )
        prior = ""
        if self._last_prediction is not None:
            prior = (
                "The previous committed prediction was "
                f"{json.dumps(self._last_prediction, sort_keys=True)} and the mechanical "
                f"update result was {json.dumps(self._last_update, sort_keys=True)}. "
            )
        if self.condition == "required":
            contract = (
                "Before answering, create or revise world_model.py. It must define "
                "parse_observation(path)->dict, predict(state, action)->dict, and "
                "update(state, action, observation_path)->dict. The controller—not your "
                "prose—will load this file, invoke predict for every legal action, and use "
                "its result as the commitment. "
            )
        else:
            contract = (
                "You may create or revise world_model.py with parse_observation, predict, "
                "and update if executable reconstruction is useful. If it exists and loads, "
                "the controller will use it; otherwise your direct JSON prediction is used. "
            )
        return (
            common
            + prior
            + contract
            + "Use Python as needed. Finish with exactly one JSON object containing "
            "prediction and candidate_predictions objects."
        )

    def _execute_tool_calls(
        self,
        tool_calls: Sequence[Mapping[str, Any]],
        *,
        model_round: int,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for index, tool_call in enumerate(tool_calls):
            tool_id = str(tool_call.get("id", f"tool-{model_round}-{index}"))
            function = tool_call.get("function")
            try:
                if not isinstance(function, Mapping) or function.get("name") != "python":
                    raise ValueError("only the python tool exists")
                arguments = function.get("arguments", "{}")
                parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
                if not isinstance(parsed, Mapping) or not isinstance(parsed.get("code"), str):
                    raise ValueError("python tool requires a string code field")
                result = self.workspace.execute(parsed["code"])
                self.python_calls += 1
                output = result.model_text()
            except (json.JSONDecodeError, ValueError) as error:
                output = json.dumps({"ok": False, "stderr": str(error)})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "name": "python",
                    "content": output,
                }
            )
        return messages


@dataclass(frozen=True)
class _ArtifactResult:
    loaded: bool
    sha256: str
    entrypoints: tuple[str, ...]
    candidate_predictions: dict[str, dict[str, Any]]


def _artifact_script(legal_actions: tuple[int, ...]) -> str:
    return f'''\
from pathlib import Path
import contextlib, hashlib, io, json
_path = Path("world_model.py")
_payload = {{"exists": _path.is_file()}}
if _path.is_file():
    _source = _path.read_bytes()
    _ns = {{"__name__": "__reconstruction__", "__file__": str(_path)}}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        exec(compile(_source, str(_path), "exec"), _ns, _ns)
    _entrypoints = [name for name in ("parse_observation", "predict", "update") if callable(_ns.get(name))]
    _loaded = len(_entrypoints) == 3
    _payload.update({{"artifact_sha256": hashlib.sha256(_source).hexdigest(), "entrypoints": _entrypoints, "loaded": _loaded}})
    if _loaded:
        _state_path = Path("reconstruction-state.json")
        if _state_path.is_file():
            _state = json.loads(_state_path.read_text(encoding="utf-8"))
        else:
            _state = _ns["parse_observation"]("current_observation.json")
            _state_path.write_text(json.dumps(_state, separators=(",", ":"), sort_keys=True), encoding="utf-8")
        _predictions = {{str(action): _ns["predict"](_state, action) for action in {list(legal_actions)!r}}}
        _payload["candidate_predictions"] = _predictions
print({_RESULT_MARKER!r} + json.dumps(_payload, separators=(",", ":"), sort_keys=True))
'''


def _update_script(recorded_action: int) -> str:
    return f'''\
from pathlib import Path
import contextlib, io, json
_model_path = Path("world_model.py")
_state_path = Path("reconstruction-state.json")
_payload = {{"updated": False, "artifact_exists": _model_path.is_file(), "state_exists": _state_path.is_file()}}
if _model_path.is_file() and _state_path.is_file():
    _ns = {{"__name__": "__reconstruction__", "__file__": str(_model_path)}}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        exec(compile(_model_path.read_bytes(), str(_model_path), "exec"), _ns, _ns)
        _state = json.loads(_state_path.read_text(encoding="utf-8"))
        _state = _ns["update"](_state, {recorded_action!r}, "current_observation.json")
    _state_path.write_text(json.dumps(_state, separators=(",", ":"), sort_keys=True), encoding="utf-8")
    _payload["updated"] = True
print({_RESULT_MARKER!r} + json.dumps(_payload, separators=(",", ":"), sort_keys=True))
'''


def _marked_payload(stdout: str) -> Mapping[str, Any] | None:
    position = stdout.rfind(_RESULT_MARKER)
    if position < 0:
        return None
    try:
        value = json.loads(stdout[position + len(_RESULT_MARKER) :].strip())
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, Mapping) else None


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item.get("text", "")
            for item in content
            if isinstance(item, Mapping) and isinstance(item.get("text"), str)
        )
    return ""


def _transport_evidence(
    source: object,
    *,
    model_call: int,
    usage: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model_call": model_call,
        "request_payload": getattr(source, "request_payload", None),
        "request_payload_sha256": getattr(source, "request_payload_sha256", None),
        "request_payload_encoding": getattr(source, "request_payload_encoding", None),
        "response_id": getattr(source, "response_id", None),
        "response_body_sha256": getattr(source, "response_body_sha256", None),
        "response_body_raw": getattr(source, "response_body_raw", None),
        "response_body_base64": getattr(source, "response_body_base64", None),
        "usage": dict(usage),
    }


def _json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as error:
        start = stripped.find("{")
        if start < 0:
            raise RuntimeError("model response does not contain a JSON object") from error
        try:
            value, _end = json.JSONDecoder().raw_decode(stripped[start:])
        except json.JSONDecodeError as nested:
            raise RuntimeError(f"model response is not valid JSON: {nested.msg}") from nested
    if not isinstance(value, dict):
        raise RuntimeError("model response JSON must be an object")
    return value
