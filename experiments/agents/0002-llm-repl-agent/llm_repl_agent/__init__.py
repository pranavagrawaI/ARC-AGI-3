"""Generic LLM + isolated Python workspace contestant."""

from .agent import LlmAgentConfig, LlmReplAgent
from .client import ChatClient, ChatResponse, OpenAICompatibleChatClient
from .observations import ColorPermutation
from .process_log import measurement_summary, ProcessLog, write_annotation_template
from .workspace import PythonWorkspace

__all__ = [
    "ChatClient",
    "ChatResponse",
    "ColorPermutation",
    "LlmAgentConfig",
    "LlmReplAgent",
    "OpenAICompatibleChatClient",
    "measurement_summary",
    "ProcessLog",
    "PythonWorkspace",
    "write_annotation_template",
]
