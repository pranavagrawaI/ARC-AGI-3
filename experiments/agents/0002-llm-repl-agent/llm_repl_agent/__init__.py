"""Generic LLM + isolated Python workspace contestant."""

from .agent import (
    EXPERIMENT_FIRST_SYSTEM_PROMPT,
    POLICY_PROMPTS,
    LlmAgentConfig,
    LlmReplAgent,
    ModelBudgetExceeded,
)
from .client import ChatClient, ChatResponse, OpenAICompatibleChatClient
from .config import (
    MODEL_PROFILES,
    OPENROUTER_PROFILES,
    ModelEnvironment,
    apply_openrouter_routing,
    model_request_options,
    resolve_model_environment,
)
from .observations import ColorPermutation
from .process_log import measurement_summary, ProcessLog, write_annotation_template
from .workspace import MODEL_OUTPUT_CHAR_LIMIT, PythonWorkspace

__all__ = [
    "ChatClient",
    "ChatResponse",
    "ColorPermutation",
    "EXPERIMENT_FIRST_SYSTEM_PROMPT",
    "LlmAgentConfig",
    "LlmReplAgent",
    "ModelBudgetExceeded",
    "ModelEnvironment",
    "MODEL_PROFILES",
    "OPENROUTER_PROFILES",
    "MODEL_OUTPUT_CHAR_LIMIT",
    "apply_openrouter_routing",
    "model_request_options",
    "OpenAICompatibleChatClient",
    "POLICY_PROMPTS",
    "measurement_summary",
    "ProcessLog",
    "PythonWorkspace",
    "resolve_model_environment",
    "write_annotation_template",
]
