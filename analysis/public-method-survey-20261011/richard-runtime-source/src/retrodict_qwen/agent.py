"""``AgentClient`` for the Retrodict runner: prompts and tool rounds through ChatSession."""

from __future__ import annotations

from .runner import AgentReply
from .session import ChatSession, hit_output_token_limit, last_input_tokens


class QwenAgentClient:
    def __init__(self, session: ChatSession) -> None:
        self.session = session

    async def invoke(self, prompt, resume_from, *, tools=True):
        result = await self.session.run(prompt, resume_from=resume_from, tools=tools)
        usage = result.usage
        return AgentReply(
            text=result.text,
            resume_state=result.resume_state,
            input_tokens=usage.input_tokens,
            cached_tokens=usage.cached_tokens,
            output_tokens=usage.output_tokens,
            model_requests=usage.model_requests,
            stop_reason=result.stop_reason,
            truncated=hit_output_token_limit(result.responses),
            last_context_tokens=last_input_tokens(result.responses),
            budget_notices=result.budget_notices,
            length_cuts=result.length_cuts,
            error_requests=result.error_requests,
        )
