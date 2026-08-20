# tycho-nooa-llm-trace-ar25-20260819T155638Z-RESULT

Status: VALID as a defect finding; VOID for its stated question.
Run: runs/tycho-nooa-llm-trace-ar25-20260819T155638Z
Outcome: 0 actions of 20. ProtocolError at turn 0 — "No action was committed
within 25 native tool steps". 26 LLM calls, $0.10, 3 of 26 emitted a tool call.

The card asked whether the model reasons and how close to 131072 it runs. The
first is answered (yes, unasked); the second is not, because the run never got
past turn 0. The trace built to answer them found a regression instead, which
is the more valuable result: without it the failure is a bare crash, and with
it the cause is legible from the artefact alone.

## What happened

muse-glimmer puts every token on the reasoning channel: `content` was empty on
all 26 calls, and completion_tokens ran ~10 above reasoning_tokens throughout.
The uncommitted `retain_assistant_turn` kept the `LLMOutput` event whenever
`content or reasoning` was non-empty. Reasoning was always non-empty, so the
"drop the empty case" branch its own docstring described never fired, and every
step appended `{"role": "assistant", "content": ""}` ahead of the tool-call
message (`context_blocks/formatter.py::OpenAIProviderFormatter.format`).

The model diagnosed its own history, at call 22, without being asked:
"the assistant messages show error content from system, not tool output. That
means the assistant's previous message was not a valid tool call ... perhaps my
attempts to send tool calls are being interpreted as prose because the tool call
syntax is wrong". It then spent the remaining steps hunting a syntax error that
did not exist. The same change had also deleted `reasoning=` from the ToolUse
event, so reasoning stopped riding the tool-call message — the one place NOOA
supports it (`ToolCallEvent.reasoning`, formatter.py:445) — and instead got a
message of its own.

## What was ruled out

- The reasoning flag. Direct A/B on one client, three trials each: tool calls
  returned 3/3 with it on and 3/3 with it off. It also disproved the premise
  written into `llm.py`: this provider returns ~200 characters of reasoning
  unasked, and the flag raises that to ~340. It buys effort, not visibility.
- ARC_LLM_CONTEXT_WINDOW. The prompt peaked at 14,984 tokens, 11% of the window;
  nothing in NOOA's context defence could have fired.

An attempted three-config isolation (neither flag / context only / both) was
INVALID and its result should not be cited: litellm calls `load_dotenv()` at
import (`litellm/__init__.py:27`), so unsetting the variables in the shell did
not unset them in the worker — all three configs ran as "both". The same import
made `tests/test_native_flow.py` depend on the developer's own .env; the test
now deletes the flag itself.

## Fix and evidence

`retain_assistant_turn` now keeps the assistant event for prose only; reasoning
replays on `ToolCallEvent.reasoning` in all three strategies. Same game, same
seed, 3-action budget:

    before   26 calls    3 emitted a tool call    0 actions (crash)
    after    30 calls   30 emitted a tool call    3 of 3 actions

Regression covered by tests/test_history.py.

## Reading

A model that returns nothing on `content` is not exotic, and NOOA's formatter
renders `content or ""` without complaint, so nothing upstream refuses this
shape. The defect was invisible to 52 passing tests because every fake response
in the suite carries prose. Any future history rewriter should be exercised
against a reasoning-only response, not only a prose one.
