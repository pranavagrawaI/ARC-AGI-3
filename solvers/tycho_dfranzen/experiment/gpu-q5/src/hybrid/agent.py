"""Canonical agent with candidate context, role and sampling configuration."""
from __future__ import annotations

from tycho_nooa.agent import TychoNooaAgent
from tycho_nooa.strategy import NativeToolConfig
from tycho_nooa.scribe import ScribeConfig
from tycho_nooa.builder import BuilderConfig
from hybrid import policy
from hybrid.deadlines import RequestedBuilderConfig

class HybridAgent(TychoNooaAgent):
    def __init__(self, **kwargs):
        kwargs.update(strategy_config=NativeToolConfig(
            grid_keep=1_000_000, history_keep_turns=1_000_000,
            max_tokens=12*1024, max_tool_steps=25,
            max_context_value_bytes=12*1024, temperature=0.7,
        ), scribe_config=ScribeConfig(max_tokens=12*1024, max_context_value_bytes=12*1024, temperature=0.7),
            builder_config=BuilderConfig(max_tokens=12*1024, temperature=0.7),
            truncation=policy.unbounded_render(),
            deadline_config=RequestedBuilderConfig())
        super().__init__(**kwargs)
