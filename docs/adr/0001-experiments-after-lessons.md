# Experiments After Lessons

Accepted. Lesson 0049 is the last default lesson-format agent artifact; new integrated agent work moves to `/experiments` as isolated experiment implementations and logs. The lesson format was useful for pedagogy, but the next phase needs comparable, plug-and-play agent experiments with shared harness code, explicit admissibility boundaries, and per-agent research logs; `/teach` remains as historical material unless the project explicitly returns to lesson mode.

## Consequences

- Shared experiment infrastructure lives under `/experiments/harness`.
- Agent implementations live under `/experiments/agents/<number>-<agent-name>`.
- Experiment artifacts may live under `/experiments/artifacts/<number>-<agent-name>` when needed.
- Existing `/teach` files should be left alone unless the project deliberately resumes lesson work.
