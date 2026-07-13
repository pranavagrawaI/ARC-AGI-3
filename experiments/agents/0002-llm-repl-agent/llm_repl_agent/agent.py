"""Strong simple LLM policy with one generic persistent Python tool."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence

from arcengine import GameAction

from experiments.harness import Decision, Observation, Transition

from .client import ChatClient, ChatResponse
from .conversation import RollingConversation
from .observations import (
    ColorPermutation,
    ModelObservation,
    ObservationStore,
)
from .process_log import ProcessLog
from .workspace import PythonResult, PythonWorkspace


SYSTEM_PROMPT = """You control an unfamiliar interactive visual system for this run only.

At each step you receive the current rendered image, public status metadata, and the legal
action IDs. A persistent isolated Python workspace is available through the python tool.
It contains current_observation.json, current_frame.png, all earlier public observations,
and public_history.jsonl. Use it freely for inspection, experiments, search, programs,
notes, and bookkeeping. The workspace starts empty on every independent run.

Learn from visible action/observation transitions. Do not assume that an action has a
particular meaning before evidence supports it. Decide for yourself what representations,
algorithms, or plans are useful. There is no required internal vocabulary or planning
method.

When ready to act, respond with exactly one JSON object and no surrounding prose:
{"action": <legal integer action id>, "data": <object or null>, "reason": <brief string>}
For an action whose public requirement includes x and y, data must contain integer x and y
coordinates inside the displayed frame. Use the python tool instead of returning the JSON
object whenever more analysis is useful.
"""


PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "python",
        "description": (
            "Execute Python 3 in the persistent isolated workspace. State and files persist "
            "for this run; the repository, game implementation, network, and prior runs are absent."
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


@dataclass(frozen=True)
class LlmAgentConfig:
    max_context_turns: int = 6
    max_model_rounds_per_action: int = 10
    max_decision_repairs: int = 2
    max_model_calls: int | None = None
    max_reported_tokens: int | None = None


class DecisionFormatError(ValueError):
    pass


class ModelBudgetExceeded(RuntimeError):
    pass


class LlmReplAgent:
    def __init__(
        self,
        *,
        client: ChatClient,
        workspace: PythonWorkspace,
        process_log: ProcessLog,
        color_permutation: ColorPermutation | None = None,
        config: LlmAgentConfig | None = None,
    ) -> None:
        self.client = client
        self.workspace = workspace
        self.process_log = process_log
        self.config = config or LlmAgentConfig()
        self.store = ObservationStore(
            workspace.root,
            colors=color_permutation or ColorPermutation.identity(),
        )
        self.conversation = RollingConversation(
            system_message={"role": "system", "content": SYSTEM_PROMPT},
            max_turns=self.config.max_context_turns,
        )
        self.actions = 0
        self._current: ModelObservation | None = None
        self._pending_before: ModelObservation | None = None
        self._history: list[dict[str, Any]] = []
        self._usage: dict[str, int] = {}
        self._reported_tokens_total = 0
        self._model_calls = 0
        self._workspace_evidence = workspace.evidence_state()

    def act(self, observation: Observation) -> Decision:
        current = self._observation(observation)
        self._pending_before = current
        turn: list[dict[str, Any]] = [self._observation_message(current)]
        repairs = 0

        for model_round in range(1, self.config.max_model_rounds_per_action + 1):
            messages = self.conversation.messages() + turn
            response = self._complete(messages=messages, tools=(PYTHON_TOOL,))
            self.process_log.write(
                "model_response",
                action_step=self.actions + 1,
                model_round=model_round,
                request_messages=messages,
                response=response.message,
                response_id=response.response_id,
                usage=response.usage,
            )
            assistant = _assistant_message(response.message)
            turn.append(assistant)
            tool_calls = assistant.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                turn.extend(self._execute_tool_calls(
                    tool_calls,
                    action_step=self.actions + 1,
                    model_round=model_round,
                ))
                continue

            try:
                decision = self._decision_from_response(
                    response,
                    observation=observation,
                    repairs=repairs,
                )
            except DecisionFormatError as error:
                repairs += 1
                self.process_log.write(
                    "decision_repair",
                    action_step=self.actions + 1,
                    model_round=model_round,
                    repair=repairs,
                    error=str(error),
                )
                if repairs > self.config.max_decision_repairs:
                    raise
                turn.append({
                    "role": "user",
                    "content": (
                        f"Your proposed action was invalid: {error}. Return a corrected JSON "
                        "action object using the current legal actions and coordinate bounds."
                    ),
                })
                continue

            evicted = self.conversation.append(step=self.actions + 1, messages=turn)
            if evicted:
                self.process_log.write(
                    "context_eviction",
                    action_step=self.actions + 1,
                    evicted_steps=list(evicted),
                    retained_steps=list(self.conversation.retained_steps),
                )
            self.process_log.write(
                "decision",
                action_step=self.actions + 1,
                action=decision.action,
                data=decision.data,
                reasoning=decision.reasoning,
            )
            return decision

        raise RuntimeError(
            f"model did not produce an action within {self.config.max_model_rounds_per_action} rounds"
        )

    def observe(self, transition: Transition) -> None:
        before = self._pending_before or self._observation(transition.before)
        self.actions += 1
        after = self.store.write_observation(transition.after, step=self.actions)
        history = self.store.append_transition(
            transition,
            step=self.actions,
            before=before,
            after=after,
        )
        self._history.append(history)
        self._current = after
        self._pending_before = None
        self.process_log.write("transition", **history)

    def finalize(self, observation: Observation) -> Mapping[str, Any] | None:
        """Collect a retrospective after control has ended; this cannot affect score."""

        current = self._observation(observation)
        turn: list[dict[str, Any]] = [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "The run has ended. Retrospectively summarize what you inferred. Return "
                        "one JSON object with keys summary, predictive_description, reusable_procedure, "
                        "final_plan, and uncertainties. This report is measurement only."
                    ),
                },
                {"type": "image_url", "image_url": {"url": current.png_data_url}},
            ],
        }]
        request_messages = self.conversation.messages() + turn
        budget_reason = self._budget_reason()
        if budget_reason is not None:
            self.process_log.write(
                "retrospective_skipped",
                actions=self.actions,
                reason=budget_reason,
            )
            return {"skipped": budget_reason}
        response = self._complete(messages=request_messages, tools=())
        self.process_log.write(
            "model_response",
            phase="retrospective",
            action_step=self.actions,
            model_round=1,
            request_messages=request_messages,
            response=response.message,
            response_id=response.response_id,
            usage=response.usage,
        )
        content = _message_text(response.message.get("content"))
        try:
            report = _json_object(content)
        except DecisionFormatError:
            report = {"raw_report": content}
        self.process_log.write("retrospective", actions=self.actions, report=report)
        return report

    @property
    def usage(self) -> Mapping[str, int]:
        return dict(self._usage)

    @property
    def model_calls(self) -> int:
        return self._model_calls

    def _complete(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> ChatResponse:
        budget_reason = self._budget_reason()
        if budget_reason is not None:
            self.process_log.write(
                "model_budget_exhausted",
                actions=self.actions,
                model_calls=self._model_calls,
                usage=self.usage,
                reason=budget_reason,
            )
            raise ModelBudgetExceeded(budget_reason)
        self._model_calls += 1
        response = self.client.complete(messages=messages, tools=tools)
        self._add_usage(response.usage)
        return response

    def _budget_reason(self) -> str | None:
        max_calls = self.config.max_model_calls
        if max_calls is not None and self._model_calls >= max_calls:
            return f"model-call budget exhausted ({self._model_calls}/{max_calls})"
        max_tokens = self.config.max_reported_tokens
        reported_tokens = self._reported_tokens()
        if max_tokens is not None and reported_tokens >= max_tokens:
            return (
                "reported-token budget exhausted "
                f"({reported_tokens}/{max_tokens}); the last response may cross this gate"
            )
        return None

    def _reported_tokens(self) -> int:
        return self._reported_tokens_total

    def _observation(self, observation: Observation) -> ModelObservation:
        if self._current is None:
            self._current = self.store.write_observation(observation, step=self.actions)
            self.process_log.write(
                "observation",
                action_step=self.actions,
                digest=self._current.digest,
                metadata=self._current.metadata,
            )
        return self._current

    def _observation_message(self, current: ModelObservation) -> dict[str, Any]:
        recent = self._history[-8:]
        text = json.dumps({
            "step": self.actions,
            "current": current.metadata,
            "recent_transitions": recent,
            "full_history_file": "public_history.jsonl",
            "current_observation_file": "current_observation.json",
        }, indent=2, sort_keys=True)
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": current.png_data_url}},
            ],
        }

    def _execute_tool_calls(
        self,
        tool_calls: Sequence[Mapping[str, Any]],
        *,
        action_step: int,
        model_round: int,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for index, tool_call in enumerate(tool_calls):
            tool_id = str(tool_call.get("id", f"tool-{model_round}-{index}"))
            function = tool_call.get("function")
            if not isinstance(function, Mapping) or function.get("name") != "python":
                output = PythonResult(
                    ok=False,
                    stdout="",
                    stderr="only the python tool exists",
                ).model_text()
                code = None
            else:
                arguments = function.get("arguments", "{}")
                try:
                    parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
                    if not isinstance(parsed, Mapping) or not isinstance(parsed.get("code"), str):
                        raise ValueError("python tool requires a string code field")
                    code = str(parsed["code"])
                    result = self.workspace.execute(code)
                    output = result.model_text()
                except (json.JSONDecodeError, ValueError) as error:
                    code = None
                    output = PythonResult(
                        ok=False,
                        stdout="",
                        stderr=str(error),
                    ).model_text()
            try:
                output_record = json.loads(output)
            except json.JSONDecodeError:
                output_record = {}
            self.process_log.write(
                "python_tool",
                phase="policy",
                action_step=action_step,
                model_round=model_round,
                tool_call_id=tool_id,
                code=code,
                output=output,
                output_chars=len(output),
                output_truncated=bool(output_record.get("truncated", False)),
            )
            self._log_workspace_changes(
                action_step=action_step,
                model_round=model_round,
                tool_call_id=tool_id,
            )
            messages.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "name": "python",
                "content": output,
            })
        return messages

    def _log_workspace_changes(
        self,
        *,
        action_step: int,
        model_round: int,
        tool_call_id: str,
    ) -> None:
        current = self.workspace.evidence_state()
        changes = []
        for path in sorted(set(self._workspace_evidence) | set(current)):
            before = self._workspace_evidence.get(path)
            after = current.get(path)
            if before == after:
                continue
            changes.append({"path": path, "before": before, "after": after})
        self.process_log.write(
            "workspace_change",
            phase="policy",
            action_step=action_step,
            model_round=model_round,
            tool_call_id=tool_call_id,
            changes=changes,
        )
        self._workspace_evidence = current

    def _decision_from_response(
        self,
        response: ChatResponse,
        *,
        observation: Observation,
        repairs: int,
    ) -> Decision:
        content = _message_text(response.message.get("content"))
        payload = _json_object(content)
        action = payload.get("action")
        if isinstance(action, bool) or not isinstance(action, int):
            raise DecisionFormatError("action must be an integer")
        if action not in observation.available_actions:
            raise DecisionFormatError(
                f"action {action} is not legal; legal actions are {observation.available_actions}"
            )
        raw_data = payload.get("data")
        if raw_data is not None and not isinstance(raw_data, Mapping):
            raise DecisionFormatError("data must be an object or null")
        data = dict(raw_data) if isinstance(raw_data, Mapping) else None
        game_action = GameAction.from_id(action)
        if game_action.is_complex():
            if data is None or not all(
                isinstance(data.get(key), int) and not isinstance(data.get(key), bool)
                for key in ("x", "y")
            ):
                raise DecisionFormatError(f"action {action} requires integer x and y data")
            height, width = observation.pixels.shape[-2:]
            if not (0 <= data["x"] < width and 0 <= data["y"] < height):
                raise DecisionFormatError(
                    f"coordinates {(data['x'], data['y'])} are outside width={width}, height={height}"
                )
        reason = payload.get("reason")
        return Decision(
            action=action,
            data=data,
            reasoning={
                "policy": "generic_llm_python_workspace",
                "reason": str(reason)[:2000] if reason is not None else "",
                "decision_repairs": repairs,
                "response_id": response.response_id,
            },
        )

    def _add_usage(self, usage: Mapping[str, Any]) -> None:
        total_tokens = usage.get("total_tokens")
        if isinstance(total_tokens, int) and not isinstance(total_tokens, bool):
            self._reported_tokens_total += total_tokens
        else:
            for key in ("prompt_tokens", "completion_tokens"):
                value = usage.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    self._reported_tokens_total += value
        for key, value in usage.items():
            if isinstance(value, int) and not isinstance(value, bool):
                self._usage[key] = self._usage.get(key, 0) + value


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, Mapping) and item.get("type") in {"text", "output_text"}
        )
    return ""


def _assistant_message(message: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only fields valid when an assistant response is sent back as input."""

    output: dict[str, Any] = {
        "role": "assistant",
        "content": message.get("content"),
    }
    if isinstance(message.get("tool_calls"), list):
        output["tool_calls"] = message["tool_calls"]
    if isinstance(message.get("reasoning_content"), str):
        output["reasoning_content"] = message["reasoning_content"]
    return output


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
            raise DecisionFormatError("response does not contain a JSON object") from error
        try:
            value, _end = json.JSONDecoder().raw_decode(stripped[start:])
        except json.JSONDecodeError as nested:
            raise DecisionFormatError(f"response is not valid JSON: {nested.msg}") from nested
    if not isinstance(value, dict):
        raise DecisionFormatError("response JSON must be an object")
    return value
