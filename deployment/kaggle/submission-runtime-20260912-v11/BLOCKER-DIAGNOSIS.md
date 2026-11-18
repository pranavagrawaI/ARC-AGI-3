# Reasoning-default blocker

The retained rerun-3 response for call 3 reports `finish_reason=length`,
19,252 prompt tokens, 16,384 completion tokens, and 8,924 reasoning tokens.
It contains no usable content or tool calls. The provider's remaining 7,460
completion tokens cannot be classified from those counters. The earlier claim
that every completion token was reasoning was incorrect. The CodeAct exception
is a generic diagnostic, not authoritative token accounting.

The copied candidate's `build_llm()` rejected disabled reasoning, and the relay
explicitly forced and required reasoning ON. That configuration contradicted
the requested routine Actor policy. In v2 the client defaults both endpoint
switches to false and rejects conflicting settings. Explicit opt-in remains
available for a future scoped deep path. The v2 relay requires both switches
OFF and points at a fresh output directory. The token ceiling is unchanged.

Before the repair, all three default-off regressions failed at the old guard.
Afterward, 12 focused tests pass, including real CompletionClient transport and
the real worker/adapter action path against a loopback endpoint with reasoning
OFF and ON. Valid null-prose tool calls remain accepted. Fixture responses are
scripted; these tests establish configuration and execution, not model quality.

Command: `solvers/tycho_nooa/.venv/bin/pytest -q deployment/kaggle/designer-player-discovery-20260908-v2/test_reasoning.py deployment/kaggle/designer-player-discovery-20260908-v2/test_discovery_fixture.py`

No paid rerun was made for this repair. Existing copied notebooks, launch
scripts, cards, and assessments describe the old candidate and require rebuilding.
LAUNCH.json is marked unready. Context compaction, scoped deep investigation,
factorial runner corrections, and refreshed packaging remain implementation work.

The earlier context-registration explanation is also unproven: the relay did
not retain the rejected request, and a summarizer request need not contain Actor
context. Replacing deprecated syntax alone does not establish a causal fix.
