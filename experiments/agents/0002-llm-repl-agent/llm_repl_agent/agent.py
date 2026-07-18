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
    OBSERVATION_FILE_CONTRACT,
    ObservationStore,
)
from .process_log import ProcessLog
from .workspace import PythonResult, PythonWorkspace


SYSTEM_PROMPT = f"""You control an unfamiliar interactive visual system for this run only.

At each step you receive the current rendered image, public status metadata, and the legal
action IDs. A persistent isolated Python workspace is available through the python tool.
It contains current_observation.json, current_frame.png, all earlier public observations,
and public_history.jsonl. {OBSERVATION_FILE_CONTRACT} Use the workspace freely for
inspection, experiments, search, programs, notes, and bookkeeping. It starts empty on
every independent run. Each request includes a machine-readable representations object
describing the grid and rendered image layouts, plus a deliberation object reporting the
current per-action model round and Python-call counts.

Learn from visible action/observation transitions. Do not assume that an action has a
particular meaning before evidence supports it. Decide for yourself what representations,
algorithms, or plans are useful. There is no required internal vocabulary or planning
method.

When ready to act, respond with exactly one JSON object and no surrounding prose:
{{"action": <legal integer action id>, "data": <object or null>, "reason": <brief string>}}
For an action whose public requirement includes x and y, data must contain integer x and y
coordinates inside the displayed frame. Use the python tool instead of returning the JSON
object whenever more analysis is useful.
"""


EXPERIMENT_FIRST_SYSTEM_PROMPT = SYSTEM_PROMPT + """

Experiment-first mode: wing it. Prioritize learning through interaction over explaining
the initial screen. Do not try to understand the whole scene before acting. Make quick
legal probes, observe their visible consequences, and revise your working model. Act as
soon as you have a reasonable probe; uncertainty is a reason to experiment, not a reason
to keep inspecting. Use Python for concise comparisons and bookkeeping when that directly
helps choose or interpret an experiment, not for exhaustive description before acting.
"""


POLICY_PROMPTS = {
    "neutral": SYSTEM_PROMPT,
    "experiment-first": EXPERIMENT_FIRST_SYSTEM_PROMPT,
}


PYTHON_CONTEXT_OMISSION = {
    "role": "user",
    "content": "Earlier Python exchanges omitted; workspace persists.",
}


ACTION_REQUIRED_MESSAGE = {
    "role": "user",
    "content": (
        "This is the final deliberation round. The Python tool is unavailable. "
        "Return exactly one legal action JSON object now."
    ),
}


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
    drop_stale_python_context: bool = False


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
        system_prompt: str = SYSTEM_PROMPT,
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
            system_message={"role": "system", "content": system_prompt},
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
        turn: list[dict[str, Any]] = [self._observation_message(
            current,
            model_round=1,
            python_calls_this_action=0,
        )]
        repairs = 0
        python_calls_this_action = 0

        for model_round in range(1, self.config.max_model_rounds_per_action + 1):
            if model_round > 1:
                turn[0] = self._observation_message(
                    current,
                    model_round=model_round,
                    python_calls_this_action=python_calls_this_action,
                )
            action_required = model_round == self.config.max_model_rounds_per_action
            if action_required:
                turn.append(ACTION_REQUIRED_MESSAGE)
                self.process_log.write(
                    "action_required",
                    action_step=self.actions + 1,
                    model_round=model_round,
                    reason="final_deliberation_round",
                )
                response, repairs = self._required_action_response(
                    turn=turn,
                    observation=observation,
                    model_round=model_round,
                    repairs=repairs,
                )
            else:
                response = self._complete_policy_response(
                    messages=self.conversation.messages() + turn,
                    tools=(PYTHON_TOOL,),
                    model_round=model_round,
                )
            assistant = _assistant_message(response.message)
            turn.append(assistant)
            tool_calls = assistant.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                tool_messages, executed_python_calls = self._execute_tool_calls(
                    tool_calls,
                    action_step=self.actions + 1,
                    model_round=model_round,
                )
                turn.extend(tool_messages)
                python_calls_this_action += executed_python_calls
                if self.config.drop_stale_python_context:
                    compacted = _latest_python_exchange(turn)
                    self._log_context_compaction(
                        before=turn,
                        after=compacted,
                        action_step=self.actions + 1,
                        model_round=model_round,
                        phase="within_action",
                    )
                    turn = compacted
                continue

            try:
                decision = self._decision_from_response(
                    response,
                    observation=observation,
                    repairs=repairs,
                )
            except DecisionFormatError as error:
                repairs = self._append_decision_repair(
                    turn=turn,
                    error=error,
                    model_round=model_round,
                    repairs=repairs,
                )
                continue

            conversation_turn = turn
            if self.config.drop_stale_python_context:
                conversation_turn = _without_python_exchanges(turn)
                self._log_context_compaction(
                    before=turn,
                    after=conversation_turn,
                    action_step=self.actions + 1,
                    model_round=model_round,
                    phase="action_complete",
                )
            evicted = self.conversation.append(
                step=self.actions + 1,
                messages=conversation_turn,
            )
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

    def _required_action_response(
        self,
        *,
        turn: list[dict[str, Any]],
        observation: Observation,
        model_round: int,
        repairs: int,
    ) -> tuple[ChatResponse, int]:
        while True:
            response = self._complete_policy_response(
                messages=self.conversation.messages() + turn,
                tools=(),
                model_round=model_round,
            )
            assistant = _assistant_message(response.message)
            tool_calls = assistant.get("tool_calls")
            try:
                if isinstance(tool_calls, list) and tool_calls:
                    raise DecisionFormatError(
                        "Python is unavailable on the final deliberation round"
                    )
                self._decision_from_response(
                    response,
                    observation=observation,
                    repairs=repairs,
                )
            except DecisionFormatError as error:
                if not tool_calls:
                    turn.append(assistant)
                repairs = self._append_decision_repair(
                    turn=turn,
                    error=error,
                    model_round=model_round,
                    repairs=repairs,
                )
                continue
            return response, repairs

    def _complete_policy_response(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model_round: int,
    ) -> ChatResponse:
        response = self._complete(messages=messages, tools=tools)
        self.process_log.write(
            "model_response",
            action_step=self.actions + 1,
            model_round=model_round,
            request_messages=messages,
            response=response.message,
            response_id=response.response_id,
            usage=response.usage,
        )
        return response

    def _append_decision_repair(
        self,
        *,
        turn: list[dict[str, Any]],
        error: DecisionFormatError,
        model_round: int,
        repairs: int,
    ) -> int:
        repairs += 1
        self.process_log.write(
            "decision_repair",
            action_step=self.actions + 1,
            model_round=model_round,
            repair=repairs,
            error=str(error),
        )
        if repairs > self.config.max_decision_repairs:
            raise error
        turn.append({
            "role": "user",
            "content": (
                f"Your proposed action was invalid: {error}. Return a corrected JSON "
                "action object using the current legal actions and coordinate bounds."
            ),
        })
        return repairs

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
                representations=self._current.representations,
            )
        return self._current

    def _observation_message(
        self,
        current: ModelObservation,
        *,
        model_round: int,
        python_calls_this_action: int,
    ) -> dict[str, Any]:
        recent = self._history[-8:]
        text = json.dumps({
            "step": self.actions,
            "deliberation": {
                "model_round": model_round,
                "max_model_rounds": self.config.max_model_rounds_per_action,
                "python_calls_this_action": python_calls_this_action,
                "remaining_rounds": self.config.max_model_rounds_per_action - model_round + 1,
            },
            "current_metadata": current.metadata,
            "representations": current.representations,
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
    ) -> tuple[list[dict[str, Any]], int]:
        messages: list[dict[str, Any]] = []
        executed_python_calls = 0
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
                    executed_python_calls += 1
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
        return messages, executed_python_calls

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

    def _log_context_compaction(
        self,
        *,
        before: list[dict[str, Any]],
        after: list[dict[str, Any]],
        action_step: int,
        model_round: int,
        phase: str,
    ) -> None:
        discarded_messages = len(before) - len(after)
        if discarded_messages <= 0:
            return
        self.process_log.write(
            "context_compaction",
            phase=phase,
            action_step=action_step,
            model_round=model_round,
            discarded_messages=discarded_messages,
            before_messages=len(before),
            after_messages=len(after),
            before_chars=len(json.dumps(before, ensure_ascii=False)),
            after_chars=len(json.dumps(after, ensure_ascii=False)),
        )

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


def _has_tool_calls(message: Mapping[str, Any]) -> bool:
    return message.get("role") == "assistant" and bool(message.get("tool_calls"))


def _latest_python_exchange(turn: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tool_call_indices = [
        index for index, message in enumerate(turn) if _has_tool_calls(message)
    ]
    if len(tool_call_indices) <= 1:
        return turn
    latest = tool_call_indices[-1]
    latest_tool_ids = {
        str(tool_call.get("id"))
        for tool_call in turn[latest].get("tool_calls", [])
        if isinstance(tool_call, Mapping)
    }
    compacted: list[dict[str, Any]] = []
    for index, message in enumerate(turn):
        if message == PYTHON_CONTEXT_OMISSION:
            continue
        if _has_tool_calls(message) and index != latest:
            continue
        if message.get("role") == "tool":
            if str(message.get("tool_call_id")) not in latest_tool_ids:
                continue
        if index == latest:
            compacted.append(dict(PYTHON_CONTEXT_OMISSION))
        compacted.append(message)
    return compacted


def _without_python_exchanges(turn: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        message
        for message in turn
        if message.get("role") != "tool"
        and not _has_tool_calls(message)
        and message != PYTHON_CONTEXT_OMISSION
    ]


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
