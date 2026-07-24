# Coding-Agent Harness Source Study

## Question

What does a real coding-agent harness do that the current ARC LLM+Python harness does not, especially around cumulative tool use, context management, persistent state, and completion?

## Scope and provenance

Primary-source checkouts inspected on 2026-07-20 and 2026-07-21:

| System | Commit | Role in comparison |
|---|---|---|
| OpenAI Codex CLI | [`1836ae0612052137d0cabaff7807ff8314cee940`](https://github.com/openai/codex/tree/1836ae0612052137d0cabaff7807ff8314cee940) | Production orchestration, context manager, compaction, world-state reinjection |
| mini-SWE-agent | [`38c01a19ed1a58dd17dd7c95010e4f69d059c777`](https://github.com/SWE-agent/mini-swe-agent/tree/38c01a19ed1a58dd17dd7c95010e4f69d059c777) | Minimal end-to-end coding-agent loop |
| Aider | [`5dc9490bb35f9729ef2c95d00a19ccd30c26339c`](https://github.com/Aider-AI/aider/tree/5dc9490bb35f9729ef2c95d00a19ccd30c26339c) | Host-computed repository map and chat-history summarization |
| Pi coding agent | [`c8c3cd499f4d35c0f9cfebfec5f4e3822411a49f`](https://github.com/earendil-works/pi/tree/c8c3cd499f4d35c0f9cfebfec5f4e3822411a49f) | User/project-scoped skills, extensions, session trees, and compaction |
| Hermes Agent | [`693d3909c86a38f01226a39f77de488bdfa777c5`](https://github.com/NousResearch/hermes-agent/tree/693d3909c86a38f01226a39f77de488bdfa777c5) | Curated cross-session memory, procedural skills, session retrieval, and post-turn review |
| Tufa Labs Duck harness | [`7652836056c59e044f093e3c13ed7438c814169e`](https://github.com/Tufalabs/duck-harness/tree/7652836056c59e044f093e3c13ed7438c814169e) | ARC-specific multimodal coding-agent loop and episode learning |

This document separates source observations from ARC transfer implications. It does not claim these implementations are mutually comparable benchmarks.

Duck provenance caveat: the pinned public snapshot differs from the initial public commit only by README formatting, but the bundled run records unpublished/dirty source commit `dd50b552...` on a `user/periodic-chat-clearing` branch, plus dirty dependency repositories. Findings below distinguish public-code mechanisms from mechanisms demonstrated by the bundled traces; the public tree is not a byte-for-byte source freeze of that run.

## Plain-language model

A coding harness does not make the model remember an entire repository in hidden state. It gives the model a loop around an authoritative external world:

```text
model request
  -> tool/command request
  -> harness executes against persistent filesystem/worktree
  -> structured bounded output returns to history
  -> model repeats until it explicitly finishes
```

The filesystem, git diff, tests, and compiler are durable task state. Conversation history helps the model navigate that state, but the task's truth does not disappear when prose is compacted.

Our ARC harness has raw observations and history files, but no compact authoritative semantic state comparable to a worktree, test result, or symbol map. The model must repeatedly reconstruct entities, action semantics, and progress from pixels.

## 1. mini-SWE-agent: the minimal real loop

### Direct observations

The complete default agent loop is small:

- `run()` initializes system and task messages, repeatedly calls `step()`, saves after every iteration, and exits on an explicit exit message: [`default.py:88-122`](https://github.com/SWE-agent/mini-swe-agent/blob/38c01a19ed1a58dd17dd7c95010e4f69d059c777/src/minisweagent/agents/default.py#L88-L122).
- `step()` is exactly `execute_actions(query())`: [`default.py:124-126`](https://github.com/SWE-agent/mini-swe-agent/blob/38c01a19ed1a58dd17dd7c95010e4f69d059c777/src/minisweagent/agents/default.py#L124-L126).
- `query()` sends the accumulated `self.messages` history and records cost/call counts: [`default.py:128-150`](https://github.com/SWE-agent/mini-swe-agent/blob/38c01a19ed1a58dd17dd7c95010e4f69d059c777/src/minisweagent/agents/default.py#L128-L150).
- `execute_actions()` executes commands in the environment and appends formatted observations: [`default.py:152-155`](https://github.com/SWE-agent/mini-swe-agent/blob/38c01a19ed1a58dd17dd7c95010e4f69d059c777/src/minisweagent/agents/default.py#L152-L155).
- The full trajectory and configuration are serialized, and optionally written after every loop iteration: [`default.py:157-188`](https://github.com/SWE-agent/mini-swe-agent/blob/38c01a19ed1a58dd17dd7c95010e4f69d059c777/src/minisweagent/agents/default.py#L157-L188).
- Limits are explicit: model calls, cost, wall time, and consecutive format errors: [`default.py:19-35`](https://github.com/SWE-agent/mini-swe-agent/blob/38c01a19ed1a58dd17dd7c95010e4f69d059c777/src/minisweagent/agents/default.py#L19-L35), [`default.py:128-145`](https://github.com/SWE-agent/mini-swe-agent/blob/38c01a19ed1a58dd17dd7c95010e4f69d059c777/src/minisweagent/agents/default.py#L128-L145).
- The prompt specifies a workflow—inspect, reproduce, edit, verify, test—and an explicit irreversible completion command: [`mini.yaml:9-42`](https://github.com/SWE-agent/mini-swe-agent/blob/38c01a19ed1a58dd17dd7c95010e4f69d059c777/src/minisweagent/config/mini.yaml#L9-L42).
- Tool output is structurally bounded to 10,000 characters using a head/tail representation rather than silently dropping the exchange: [`mini.yaml:112-128`](https://github.com/SWE-agent/mini-swe-agent/blob/38c01a19ed1a58dd17dd7c95010e4f69d059c777/src/minisweagent/config/mini.yaml#L112-L128).
- Shell-local state does not persist between commands, but filesystem mutations do. Each subprocess has a timeout and process-group cleanup: [`environments/local.py:24-43`](https://github.com/SWE-agent/mini-swe-agent/blob/38c01a19ed1a58dd17dd7c95010e4f69d059c777/src/minisweagent/environments/local.py#L24-L43), [`local.py:72-92`](https://github.com/SWE-agent/mini-swe-agent/blob/38c01a19ed1a58dd17dd7c95010e4f69d059c777/src/minisweagent/environments/local.py#L72-L92).
- Command/runtime failures become observations for the next model step. Model transport failures are retried selectively, while format errors become corrective messages and terminate after a repeated-error limit: [`models/utils/retry.py:9-25`](https://github.com/SWE-agent/mini-swe-agent/blob/38c01a19ed1a58dd17dd7c95010e4f69d059c777/src/minisweagent/models/utils/retry.py#L9-L25), [`default.py:96-117`](https://github.com/SWE-agent/mini-swe-agent/blob/38c01a19ed1a58dd17dd7c95010e4f69d059c777/src/minisweagent/agents/default.py#L96-L117).

### Mechanism

mini-SWE-agent is not sophisticated memory architecture. It works because:

1. the complete message trajectory is normally supplied;
2. each action gets an explicit observation;
3. files and edits persist outside the conversation;
4. tests provide an independent pass/fail signal;
5. the model is taught a concrete workflow and completion protocol.

### ARC implication

A small loop can be enough when the external environment is self-describing and durable. ARC is not: a raw frame does not say which entity moved, which rule is confirmed, or whether a visual transient is causal. Copying mini-SWE-agent's loop without creating an ARC equivalent of persistent worktree/test state reproduces our current failure.

## 2. Codex CLI: production orchestration and context

### Agent loop

`run_turn()` explicitly documents and implements the production loop:

- model returns tool calls or an assistant message;
- tool calls are executed and their outputs go into the next sampling request;
- an assistant-only response completes the turn.

Source: [`session/turn.rs:130-151`](https://github.com/openai/codex/blob/1836ae0612052137d0cabaff7807ff8314cee940/codex-rs/core/src/session/turn.rs#L130-L151).

The loop snapshots model-visible context, builds each request from normalized conversation history, and repeats while tool calls or pending input require follow-up: [`session/turn.rs:226-322`](https://github.com/openai/codex/blob/1836ae0612052137d0cabaff7807ff8314cee940/codex-rs/core/src/session/turn.rs#L226-L322).

`StepContext` freezes the environment readiness, instructions, runtime, and exact advertised tool set for one request. That same snapshot governs execution, preventing the model from seeing one tool/state contract and acting against another: [`session/step_context.rs:21-49`](https://github.com/openai/codex/blob/1836ae0612052137d0cabaff7807ff8314cee940/codex-rs/core/src/session/step_context.rs#L21-L49), [`session/mod.rs:2902-2945`](https://github.com/openai/codex/blob/1836ae0612052137d0cabaff7807ff8314cee940/codex-rs/core/src/session/mod.rs#L2902-L2945).

Completion is not merely “the model emitted prose.” Stop hooks can reject completion and inject continuation context: [`session/turn.rs:381-430`](https://github.com/openai/codex/blob/1836ae0612052137d0cabaff7807ff8314cee940/codex-rs/core/src/session/turn.rs#L381-L430).

### Context manager

Codex retains an ordered transcript as typed response items and estimates token use rather than keeping an arbitrary fixed number of turns: [`context_manager/history.rs:38-73`](https://github.com/openai/codex/blob/1836ae0612052137d0cabaff7807ff8314cee940/codex-rs/core/src/context_manager/history.rs#L38-L73), [`history.rs:162-187`](https://github.com/openai/codex/blob/1836ae0612052137d0cabaff7807ff8314cee940/codex-rs/core/src/context_manager/history.rs#L162-L187).

Before a prompt, history is normalized. The implementation maintains invariants that every tool call has a corresponding output and every output has its call; unsupported media are stripped deliberately: [`history.rs:325-342`](https://github.com/openai/codex/blob/1836ae0612052137d0cabaff7807ff8314cee940/codex-rs/core/src/context_manager/history.rs#L325-L342).

Tool outputs are processed through a truncation policy at ingestion, not deleted by turn count: [`history.rs:344-365`](https://github.com/openai/codex/blob/1836ae0612052137d0cabaff7807ff8314cee940/codex-rs/core/src/context_manager/history.rs#L344-L365).

### Compaction

When the token limit is reached and the model still needs a follow-up, Codex runs compaction and continues the same turn: [`session/turn.rs:342-379`](https://github.com/openai/codex/blob/1836ae0612052137d0cabaff7807ff8314cee940/codex-rs/core/src/session/turn.rs#L342-L379).

The local compaction prompt asks for a handoff containing:

- current progress and decisions;
- constraints/preferences;
- remaining work;
- critical data and references.

Source: [`prompts/templates/compact/prompt.md`](https://github.com/openai/codex/blob/1836ae0612052137d0cabaff7807ff8314cee940/codex-rs/prompts/templates/compact/prompt.md).

Compaction replaces history with a summary-bearing history and records the replacement, rather than merely deleting the oldest N turns: [`compact.rs:323-356`](https://github.com/openai/codex/blob/1836ae0612052137d0cabaff7807ff8314cee940/codex-rs/core/src/compact.rs#L323-L356).

### World state

Codex separately maintains typed model-visible world-state sections for live environment/configuration context. Each section has:

- a stable ID;
- a serializable snapshot;
- a renderer that emits the difference from the previous known snapshot.

Source: [`context/world_state/mod.rs:173-216`](https://github.com/openai/codex/blob/1836ae0612052137d0cabaff7807ff8314cee940/codex-rs/core/src/context/world_state/mod.rs#L173-L216).

The aggregate `WorldState` can snapshot itself and render full state or diffs against a persisted baseline: [`world_state/mod.rs:239-270`](https://github.com/openai/codex/blob/1836ae0612052137d0cabaff7807ff8314cee940/codex-rs/core/src/context/world_state/mod.rs#L239-L270), [`world_state/mod.rs:280-360`](https://github.com/openai/codex/blob/1836ae0612052137d0cabaff7807ff8314cee940/codex-rs/core/src/context/world_state/mod.rs#L280-L360).

The session rebuilds world state for each model step and contributes environment, instructions, permissions, collaboration, apps, plugins, and extension sections: [`session/world_state.rs:17-134`](https://github.com/openai/codex/blob/1836ae0612052137d0cabaff7807ff8314cee940/codex-rs/core/src/session/world_state.rs#L17-L134).

### Persistence and retries

Codex appends canonical rollout records to JSONL and persists turn boundaries, response items, compaction records, context, and world-state changes. Resume reconstructs context from this rollout rather than trusting a short prompt tail: [`rollout/src/recorder.rs:1-126`](https://github.com/openai/codex/blob/1836ae0612052137d0cabaff7807ff8314cee940/codex-rs/rollout/src/recorder.rs#L1-L126), [`rollout/src/policy.rs:85-181`](https://github.com/openai/codex/blob/1836ae0612052137d0cabaff7807ff8314cee940/codex-rs/rollout/src/policy.rs#L85-L181).

Model-stream retries are limited to classified retryable errors. Requests are rebuilt from current persisted history because a partial stream may already have emitted durable items or tool calls: [`session/turn.rs:1143-1210`](https://github.com/openai/codex/blob/1836ae0612052137d0cabaff7807ff8314cee940/codex-rs/core/src/session/turn.rs#L1143-L1210), [`responses_retry.rs:20-79`](https://github.com/openai/codex/blob/1836ae0612052137d0cabaff7807ff8314cee940/codex-rs/core/src/responses_retry.rs#L20-L79).

### Important limitation

Codex's built-in world state is mostly harness/environment state, not a verified database of arbitrary task beliefs. Coding-task truth remains primarily in the filesystem, git worktree, command outputs, tests, and summarized history. Its generic completion condition also does not prove a task was solved: a plain assistant response can finish a turn unless a stop hook blocks it.

### ARC implication

The transferable pattern is not “copy Codex.” It is to separate:

```text
transcript history
from live authoritative state
from compacted task progress
```

Our current ARC harness conflates these. It deletes transcript turns without producing a task-state checkpoint, and raw files are too low-level to serve as authoritative semantic state.

## 3. Aider: host-computed structural context

### Repository map

Aider does not require the model to repeatedly rediscover the repository structure through shell calls. `RepoMap`:

- parses code using tree-sitter/grep-ast;
- caches tags on disk;
- builds a ranked token-budgeted map;
- uses current filename and identifier mentions to bias relevance;
- expands the budget when no files are explicitly in chat.
- constructs a reference-to-definition graph and ranks it with PageRank;
- binary-searches the ranked symbols to fit the largest useful map inside the token budget.

Sources: [`repomap.py:42-81`](https://github.com/Aider-AI/aider/blob/5dc9490bb35f9729ef2c95d00a19ccd30c26339c/aider/repomap.py#L42-L81), [`repomap.py:103-167`](https://github.com/Aider-AI/aider/blob/5dc9490bb35f9729ef2c95d00a19ccd30c26339c/aider/repomap.py#L103-L167), [`repomap.py:365-525`](https://github.com/Aider-AI/aider/blob/5dc9490bb35f9729ef2c95d00a19ccd30c26339c/aider/repomap.py#L365-L525), [`repomap.py:629-706`](https://github.com/Aider-AI/aider/blob/5dc9490bb35f9729ef2c95d00a19ccd30c26339c/aider/repomap.py#L629-L706).

The coder derives mentioned filenames and identifiers from current messages, requests a ranked map, and falls back to broader maps if the focused map is empty: [`base_coder.py:672-748`](https://github.com/Aider-AI/aider/blob/5dc9490bb35f9729ef2c95d00a19ccd30c26339c/aider/coders/base_coder.py#L672-L748).

This map is injected separately from full editable/read-only file content: [`base_coder.py:750-815`](https://github.com/Aider-AI/aider/blob/5dc9490bb35f9729ef2c95d00a19ccd30c26339c/aider/coders/base_coder.py#L750-L815).

Each request is assembled from explicit context categories—system instructions, compacted history, repository map, read-only files, editable files, current messages, and reminders—rather than treating every old tool output as one undifferentiated transcript: [`base_coder.py:1264-1295`](https://github.com/Aider-AI/aider/blob/5dc9490bb35f9729ef2c95d00a19ccd30c26339c/aider/coders/base_coder.py#L1264-L1295), [`chat_chunks.py:5-26`](https://github.com/Aider-AI/aider/blob/5dc9490bb35f9729ef2c95d00a19ccd30c26339c/aider/coders/chat_chunks.py#L5-L26).

### History summarization

Aider divides messages into current and completed history. When completed history exceeds its token budget, it summarizes in the background and replaces only the matching old history: [`base_coder.py:1002-1046`](https://github.com/Aider-AI/aider/blob/5dc9490bb35f9729ef2c95d00a19ccd30c26339c/aider/coders/base_coder.py#L1002-L1046).

`ChatSummary` preserves a recent tail, summarizes the older head, and recursively reduces if necessary: [`history.py:27-96`](https://github.com/Aider-AI/aider/blob/5dc9490bb35f9729ef2c95d00a19ccd30c26339c/aider/history.py#L27-L96).

### ARC implication

Aider's repository map is the closest analogy to what our ARC harness lacks: a compact, host-computed structural representation that is regenerated from authoritative state and selected under a token budget.

An ARC equivalent might expose generic, non-semantic structure:

- per-color connected components;
- component bounding boxes;
- component changes and displacement candidates;
- unchanged/static geometry;
- recent transition deltas;
- status-bar changes separated from board changes.

The model would still decide which component is the player, which object is a goal, and what mechanics mean. The host would remove repeated pixel-accounting work without embedding game-specific solutions.

## 4. Pi: user-extensible, project-scoped growth

### Direct observations

Pi's durable session unit is a per-working-directory JSONL tree under `~/.pi/agent/sessions/`. Entries have IDs and parent IDs, so sessions can resume, fork, clone, and retain abandoned branches rather than flattening everything into one transcript: [`sessions.md:1-35`](https://github.com/earendil-works/pi/blob/c8c3cd499f4d35c0f9cfebfec5f4e3822411a49f/packages/coding-agent/docs/sessions.md#L1-L35), [`session-manager.ts:471-488`](https://github.com/earendil-works/pi/blob/c8c3cd499f4d35c0f9cfebfec5f4e3822411a49f/packages/coding-agent/src/core/session-manager.ts#L471-L488), [`session-manager.ts:855-884`](https://github.com/earendil-works/pi/blob/c8c3cd499f4d35c0f9cfebfec5f4e3822411a49f/packages/coding-agent/src/core/session-manager.ts#L855-L884).

Its compaction is an append-only session event with a structured summary and `firstKeptEntryId`. It preserves a recent tail, never cuts at an unpaired tool result, iteratively incorporates the previous summary, and cumulatively records read/modified files. Branch summarization carries useful context from a branch the user leaves: [`compaction.md:25-79`](https://github.com/earendil-works/pi/blob/c8c3cd499f4d35c0f9cfebfec5f4e3822411a49f/packages/coding-agent/docs/compaction.md#L25-L79), [`compaction.md:109-146`](https://github.com/earendil-works/pi/blob/c8c3cd499f4d35c0f9cfebfec5f4e3822411a49f/packages/coding-agent/docs/compaction.md#L109-L146), [`compaction.md:148-213`](https://github.com/earendil-works/pi/blob/c8c3cd499f4d35c0f9cfebfec5f4e3822411a49f/packages/coding-agent/docs/compaction.md#L148-L213).

Pi discovers global and project skills from `~/.pi/agent/skills/`, `~/.agents/skills/`, `.pi/skills/`, and `.agents/skills/`. Only skill names/descriptions are always injected; the model reads full `SKILL.md` instructions on demand: [`skills.md:24-41`](https://github.com/earendil-works/pi/blob/c8c3cd499f4d35c0f9cfebfec5f4e3822411a49f/packages/coding-agent/docs/skills.md#L24-L41), [`skills.md:64-71`](https://github.com/earendil-works/pi/blob/c8c3cd499f4d35c0f9cfebfec5f4e3822411a49f/packages/coding-agent/docs/skills.md#L64-L71).

It also discovers `AGENTS.md` or `CLAUDE.md` context files from the working-directory hierarchy and supports executable extensions with lifecycle/tool hooks: [`resource-loader.ts:67-79`](https://github.com/earendil-works/pi/blob/c8c3cd499f4d35c0f9cfebfec5f4e3822411a49f/packages/coding-agent/src/core/resource-loader.ts#L67-L79), [`extensions.md:1-52`](https://github.com/earendil-works/pi/blob/c8c3cd499f4d35c0f9cfebfec5f4e3822411a49f/packages/coding-agent/docs/extensions.md#L1-L52).

Packages can bundle extensions, skills, prompt templates, and themes, installed globally or project-locally: [`packages.md:1-43`](https://github.com/earendil-works/pi/blob/c8c3cd499f4d35c0f9cfebfec5f4e3822411a49f/packages/coding-agent/docs/packages.md#L1-L43), [`packages.md:116-169`](https://github.com/earendil-works/pi/blob/c8c3cd499f4d35c0f9cfebfec5f4e3822411a49f/packages/coding-agent/docs/packages.md#L116-L169).

### What “grows” actually means

Pi has strong **user-extensible capability growth** and durable project/session context. Users or the model can author files, skills, prompts, and extensions that later runs rediscover. However, the inspected core does not contain a Hermes-style bounded user-profile memory automatically reviewed and injected across every session. Session retention and compaction are not themselves semantic learning.

### ARC implication

The transferable Pi mechanism is an append-only episode tree with typed checkpoint events and a project-local library of reusable, game-agnostic analysis skills. We should not add arbitrary executable plugins during a benchmark run: that would expand the trusted computing base and make contestant capability hard to attribute.

## 5. Hermes: explicit cross-session learning substrate

### Direct observations

Hermes separates durable knowledge by purpose:

- `MEMORY.md` stores compact environment/convention/tool facts;
- `USER.md` stores user preferences and expectations;
- both are bounded, sanitized, and injected as a frozen system-prompt snapshot at session start.

Source: [`memory.md:7-24`](https://github.com/NousResearch/hermes-agent/blob/693d3909c86a38f01226a39f77de488bdfa777c5/website/docs/user-guide/features/memory.md#L7-L24), [`memory_tool.py:1-11`](https://github.com/NousResearch/hermes-agent/blob/693d3909c86a38f01226a39f77de488bdfa777c5/tools/memory_tool.py#L1-L11), [`memory_tool.py:123-212`](https://github.com/NousResearch/hermes-agent/blob/693d3909c86a38f01226a39f77de488bdfa777c5/tools/memory_tool.py#L123-L212).

Hermes stores full sessions in SQLite and exposes FTS5-backed retrieval of actual past messages. This is recall, not automatic prompt injection: the agent searches when relevant: [`memory.md:181-188`](https://github.com/NousResearch/hermes-agent/blob/693d3909c86a38f01226a39f77de488bdfa777c5/website/docs/user-guide/features/memory.md#L181-L188), [`session_search_tool.py:619-652`](https://github.com/NousResearch/hermes-agent/blob/693d3909c86a38f01226a39f77de488bdfa777c5/tools/session_search_tool.py#L619-L652).

Procedures live separately as `SKILL.md` packages. The system prompt contains compact skill metadata; full instructions and support files are loaded on demand. This is progressive disclosure rather than always injecting the entire learned library: [`skills.md:7-13`](https://github.com/NousResearch/hermes-agent/blob/693d3909c86a38f01226a39f77de488bdfa777c5/website/docs/user-guide/features/skills.md#L7-L13), [`skills.md:132-140`](https://github.com/NousResearch/hermes-agent/blob/693d3909c86a38f01226a39f77de488bdfa777c5/website/docs/user-guide/features/skills.md#L132-L140), [`skills_tool.py:786-845`](https://github.com/NousResearch/hermes-agent/blob/693d3909c86a38f01226a39f77de488bdfa777c5/tools/skills_tool.py#L786-L845).

The strongest literal learning mechanism is optional post-turn background review. A persistence-isolated fork receives the completed conversation, is restricted to memory/skill management tools, and decides whether to add or patch durable knowledge. The main conversation and prompt cache are not modified by the fork: [`background_review.py:1-16`](https://github.com/NousResearch/hermes-agent/blob/693d3909c86a38f01226a39f77de488bdfa777c5/agent/background_review.py#L1-L16), [`background_review.py:166-205`](https://github.com/NousResearch/hermes-agent/blob/693d3909c86a38f01226a39f77de488bdfa777c5/agent/background_review.py#L166-L205).

The trigger can be automatic, but the mutation is still an ordinary, visible model tool call. Defaults schedule memory review every ten user turns and skill review after ten agent-loop iterations without a skill mutation; write-approval settings can stage proposed memory/skill changes for explicit acceptance rather than applying them immediately: [`turn_context.py:458-501`](https://github.com/NousResearch/hermes-agent/blob/693d3909c86a38f01226a39f77de488bdfa777c5/agent/turn_context.py#L458-L501), [`turn_finalizer.py:575-601`](https://github.com/NousResearch/hermes-agent/blob/693d3909c86a38f01226a39f77de488bdfa777c5/agent/turn_finalizer.py#L575-L601), [`background_review.py:617-860`](https://github.com/NousResearch/hermes-agent/blob/693d3909c86a38f01226a39f77de488bdfa777c5/agent/background_review.py#L617-L860).

Memory written during a session is durable immediately but does not mutate the already frozen prompt snapshot; it becomes model-visible after a new session or prompt rebuild. Hermes can optionally use an external memory provider for automatic semantic prefetch and completed-turn ingestion, but semantic extraction or knowledge graphs in that path are provider capabilities, not properties of the built-in Markdown memory: [`memory_manager.py:515-684`](https://github.com/NousResearch/hermes-agent/blob/693d3909c86a38f01226a39f77de488bdfa777c5/agent/memory_manager.py#L515-L684).

Long-session context compression is a separate mechanism. It protects head/tail context, iteratively updates a structured historical summary, and explicitly marks compacted tasks as reference-only so stale work is not reactivated: [`context_compressor.py:1-17`](https://github.com/NousResearch/hermes-agent/blob/693d3909c86a38f01226a39f77de488bdfa777c5/agent/context_compressor.py#L1-L17), [`context_compressor.py:81-115`](https://github.com/NousResearch/hermes-agent/blob/693d3909c86a38f01226a39f77de488bdfa777c5/agent/context_compressor.py#L81-L115).

### What “grows” actually means

Hermes does implement cross-session learning, but not weight updates. It writes model-curated state into three different external substrates:

```text
stable user/environment facts -> bounded memory files -> automatic frozen injection
reusable procedures          -> skill files          -> metadata discovery + on-demand load
past task detail              -> SQLite sessions      -> explicit FTS retrieval
```

This separation is valuable because it avoids making all prior conversation equally authoritative. Its main risk is self-poisoning: model-authored memory or skills can encode a mistaken conclusion. Bounds, sanitization, provenance, and user inspectability reduce but do not eliminate that risk.

### ARC implication

Use the separation, not the personalization feature wholesale:

- objective episode facts belong in host-verified state;
- model mechanic hypotheses belong in a bounded episode notebook with evidence IDs;
- generic reusable analysis procedures may live as frozen benchmark-versioned skills;
- previous game traces should be retrieved only under an explicitly declared experimental protocol, never silently, because cross-game recall can become evaluation leakage.

An autonomous post-game curator must not modify the active benchmark harness between evaluation games. It may propose candidate skills offline; independent acceptance and version freezing are required before reuse.

## 6. Tufa Labs Duck: the ARC-specific implementation

### Direct observations

Duck calls the model a coding agent and exposes one `python` tool. Python receives structured, read-only episode objects—`current_frame`, `previous_frame`, `history`, `transitions`, `last_transition`, legal actions, and an `action(actions)` callback. The prompt explicitly tells the model to inspect evidence, revise its world model, program searches/scorers, and execute the chosen action from Python: [`tool_agent.py:1161-1256`](https://github.com/Tufalabs/duck-harness/blob/7652836056c59e044f093e3c13ed7438c814169e/ARC3-Inference/inference/agent/tool_agent.py#L1161-L1256), [`python_tool_sandbox.py:1-39`](https://github.com/Tufalabs/duck-harness/blob/7652836056c59e044f093e3c13ed7438c814169e/ARC3-Inference/inference/agent/python_tool_sandbox.py#L1-L39).

The runtime state is an authoritative JSON file with immutable frames and action/frame history. Frames provide both ASCII and segmentation views; transitions are derived from adjacent history entries: [`runtime_state.py:15-45`](https://github.com/Tufalabs/duck-harness/blob/7652836056c59e044f093e3c13ed7438c814169e/ARC3-Inference/inference/agent/runtime_state.py#L15-L45), [`runtime_state.py:108-135`](https://github.com/Tufalabs/duck-harness/blob/7652836056c59e044f093e3c13ed7438c814169e/ARC3-Inference/inference/agent/runtime_state.py#L108-L135).

The Python snippet is deliberately ephemeral, but every call reloads current runtime state. This is the coding-agent pattern: persistent artifacts/state, disposable interpreter process: [`tool_agent.py:1258-1279`](https://github.com/Tufalabs/duck-harness/blob/7652836056c59e044f093e3c13ed7438c814169e/ARC3-Inference/inference/agent/tool_agent.py#L1258-L1279), [`tool_agent.py:1453-1491`](https://github.com/Tufalabs/duck-harness/blob/7652836056c59e044f093e3c13ed7438c814169e/ARC3-Inference/inference/agent/tool_agent.py#L1453-L1491).

Duck extracts model-authored “scientist notes” into explicit fields:

```text
world model
goal model
action model
recent findings
open questions
plan
cross-level notes
```

These fields are reinjected on subsequent turns with an instruction to revise them when current frames/history contradict them. Most fields are cleared on a level transition; cross-level notes survive: [`tool_agent.py:278-305`](https://github.com/Tufalabs/duck-harness/blob/7652836056c59e044f093e3c13ed7438c814169e/ARC3-Inference/inference/agent/tool_agent.py#L278-L305), [`tool_agent.py:1105-1145`](https://github.com/Tufalabs/duck-harness/blob/7652836056c59e044f093e3c13ed7438c814169e/ARC3-Inference/inference/agent/tool_agent.py#L1105-L1145).

The tool loop persists paired assistant/tool messages, trims whole older history blocks against a token budget, recovers malformed text-embedded tool calls, retries context overflow after further trimming, and preserves prior history if a request fails: [`tool_agent.py:1680-1704`](https://github.com/Tufalabs/duck-harness/blob/7652836056c59e044f093e3c13ed7438c814169e/ARC3-Inference/inference/agent/tool_agent.py#L1680-L1704), [`tool_agent.py:1754-1853`](https://github.com/Tufalabs/duck-harness/blob/7652836056c59e044f093e3c13ed7438c814169e/ARC3-Inference/inference/agent/tool_agent.py#L1754-L1853), [`tool_agent.py:1887-1977`](https://github.com/Tufalabs/duck-harness/blob/7652836056c59e044f093e3c13ed7438c814169e/ARC3-Inference/inference/agent/tool_agent.py#L1887-L1977).

Actions are validated and may be batched, but execution stops early on invalid action, level completion, game over, or run completion. Environment state owns terminal truth: [`solver.py:588-660`](https://github.com/Tufalabs/duck-harness/blob/7652836056c59e044f093e3c13ed7438c814169e/ARC3-Inference/inference/framework/solver.py#L588-L660).

Duck fixes action semantics in the harness—`ACTION1→UP`, `ACTION2→DOWN`, `ACTION3→LEFT`, `ACTION4→RIGHT`, `ACTION5→SPACE`, and `ACTION6→MOUSE`—and exposes the names to the model: [`action_names.py:7-31`](https://github.com/Tufalabs/duck-harness/blob/7652836056c59e044f093e3c13ed7438c814169e/ARC3-Inference/inference/agent/action_names.py#L7-L31). This removes an unnecessary inference problem that our current harness left to Gemma.

The bundled traces confirm that the semantic notebook and context-overflow recovery executed in practice, not merely in tests: [`wa30 transcript:486-492`](https://github.com/Tufalabs/duck-harness/blob/7652836056c59e044f093e3c13ed7438c814169e/example-run/transcripts/wa30-ee6fef47_p0.txt#L486-L492), [`bp35 transcript:1879-1883`](https://github.com/Tufalabs/duck-harness/blob/7652836056c59e044f093e3c13ed7438c814169e/example-run/transcripts/bp35-0a0ad940_p16.txt#L1879-L1883). The run provenance also reports unlimited history, whereas public source caps persistent history at 30 assistant turns; therefore that cap must not be attributed to the bundled run: [`git_status.txt:1-3`](https://github.com/Tufalabs/duck-harness/blob/7652836056c59e044f093e3c13ed7438c814169e/example-run/git_status.txt#L1-L3).

### What “grows” actually means

Duck learns **within one ARC run**:

```text
objective frame/action history
+ derived transitions and segmentation
+ model-authored compact world/action/goal model
+ cross-level notes
```

It resets conversation history and summarized knowledge when the runtime directory changes: [`tool_agent.py:975-984`](https://github.com/Tufalabs/duck-harness/blob/7652836056c59e044f093e3c13ed7438c814169e/ARC3-Inference/inference/agent/tool_agent.py#L975-L984). It is not a user-personalization or cross-game learning system.

### ARC implication

Duck directly validates most of the proposed ownership seam. It goes further than our current harness by exposing structured frame history and transitions and explicitly carrying a model-authored world model. The remaining weakness is that objective and subjective state are both transported largely through prompt text, and old raw history blocks are dropped rather than semantically compacted. Our replay prototype should therefore start from Duck's actual pattern, not design an unrelated symbolic solver:

```text
Duck-compatible structured frames/transitions
+ evidence-addressed scientist notebook
+ immutable request checkpoint IDs
+ token-budgeted semantic checkpoint
```

Before those refinements, make the cheaper interface correction Duck already demonstrates: expose immutable semantic action names instead of asking the model to rediscover and remember numeric action IDs.

## Cross-system mechanism table

| Mechanism | mini-SWE | Codex | Aider | Pi | Hermes | Duck | Current ARC |
|---|---|---|---|---|---|---|---|
| Durable task authority | Worktree | Worktree + typed runtime | Worktree | Worktree + session tree | Files + stores + session DB | Runtime frames/history | Raw files |
| Cross-session user memory | No | No | No | No built-in profile store | Bounded `USER.md`/`MEMORY.md` | No | No |
| Reusable procedures | Prompt | Instructions | Conventions | Skills/extensions/packages | Skills/support files | Frozen ARC prompt | Prompt only |
| Retrieval/injection | Full history | Normalized history + world-state diffs | Ranked repo map | Skills metadata + on-demand files | Memory auto-injected; skills/search on demand | Current frame/history + notebook | Six-turn tail |
| Compaction | None normally | Structured handoff | Old-head summary | Append-only compaction + branch summary | Structured protected head/tail | Drops old paired blocks; retains notebook | Deletion only |
| Model-authored durable beliefs | No dedicated store | Summary only | Summary only | Session/extension-defined | Memories/skills, curated | Scientist notebook, episode-local | None |
| Objective transition representation | Command outputs | Diffs/tool outputs | Worktree/map | Command outputs | Tool outputs | Frames + transitions + segmentation | Pixel hashes/diffs only |
| Terminal authority | Submit command | Model + stop hooks | Coder/user flow | Agent/user flow | Agent/user flow | ARC environment | ARC environment |

## Main conclusion

The current ARC harness copied the most superficial feature of coding agents—a generic interpreter—but omitted the conditions that make that interpreter cumulative:

1. **Authoritative task state:** coding agents can always re-read files, diffs, and tests. ARC needs a compact regenerated scene/transition representation plus a separate model-authored hypothesis state.
2. **Token-aware compaction:** production agents summarize progress and reinject live state; they do not simply keep six turns and delete the rest.
3. **Bounded but retained tool observations:** tool outputs are truncated structurally while preserving call/output integrity.
4. **A concrete control loop:** inspect, change/probe, verify, update state, then continue or finish.
5. **Independent progress signals:** tests make code-agent progress falsifiable. ARC needs explicit expected-versus-observed transition checks and completion metadata.
6. **Request-scoped consistency:** every model call must record one immutable state/image version, one tool schema, and one belief/checkpoint version; the resulting tool calls and action must execute against that contract.
7. **Typed failure semantics:** transport retries, format repairs, Python failures, illegal actions, and semantic contradictions are different events. Only idempotent transport failures should be retried invisibly.

## Revised architecture direction

Do not immediately build a large symbolic solver. Introduce three deep modules behind small interfaces:

### `SceneMap.current(observation) -> SceneSnapshot`

Host-computed generic components, geometry, and display/board separation. Regenerated from authoritative pixels; no player/goal semantics.

### `TransitionMap.compare(before, action, after) -> TransitionFacts`

Objective component additions, removals, movements, and changed regions. This is the ARC analogue of command output or git diff.

### `EpisodeCheckpoint.compile(history, scene, transitions, model_state) -> ModelContext`

Token-budgeted context containing:

- current objective state;
- verified transition facts;
- model hypotheses with evidence references;
- current plan and expected next observation;
- unresolved questions;
- recent raw tail.

The model should update hypotheses; the host should own objective scene/transition facts. Conversation compaction must never erase either.

Each compiled context receives a stable checkpoint ID. Request payloads, Python calls, and the selected environment action must cite that ID, making stale-state execution visible in the trace.

## Cheapest falsifiable prototype

Use the captured Gemma `ls20` trajectory—no paid model call—to prove:

1. `SceneMap` tracks the color-12 component across all 44 frames;
2. `TransitionMap` derives the fixed displacement signatures for actions 1–4 when movement occurs and marks blocked actions separately;
3. `EpisodeCheckpoint` retains those facts after synthetic six-turn eviction;
4. request-context size is lower than the current repeated raw-history requests;
5. no game-specific color, action, player, or goal constant appears in implementation.

Only after those replay gates pass should a pinned-provider Gemma run test whether the model uses the improved harness.
