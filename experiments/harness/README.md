# Public-interface experiment harness

This harness keeps an Admissible Contestant behind a deliberately narrow
boundary. The runner owns the ARC environment and converts each response into a
copied, read-only `Observation` containing only rendered pixels, available action
IDs, level progress, terminal state, and the public reset flag.

An agent implements two methods:

```python
def act(observation: Observation) -> Decision: ...
def observe(transition: Transition) -> None: ...
```

Run it with:

```python
with JsonlTraceWriter(trace_path) as trace:
    result = run_episode(env, agent, max_actions=500, trace=trace)
```

The runner rejects actions absent from the current public action set. Its JSONL
trace starts with the initial observation, records each action and resulting
observation once, then writes the stopping condition. It contains no private game
state or oracle labels. `Decision.data` carries coordinates or other public input
required by parameterized actions; the runner validates it against the selected
ARC action before execution.

## ARC-AGI-3 contract mapping

This harness mirrors the public ARC-AGI-3 frame/action contract used by the local
`arc_agi` toolkit:

- `Observation.pixels` is copied from the public frame grid. ARC grids are at most
  64×64 and contain integer values 0–15.
- `Observation.available_actions` is the public action list. Valid ARC action IDs
  are `0` (`RESET`), `1`–`5`, `6` (coordinate action with `x,y` data), and `7`.
- `Observation.levels_completed`, `state`, and `full_reset` are copied from public
  frame metadata.
- The runner stops on public terminal states `WIN` and `GAME_OVER`.
- `Decision.action` must be present in `available_actions`; `Decision.data` is
  required and validated for `ACTION6`.

The experiment harness deliberately uses:

```python
act(observation) -> Decision
observe(transition) -> None
```

instead of the ARC-AGI-3-Agents repository's:

```python
is_done(frames, latest_frame)
choose_action(frames, latest_frame)
```

The information boundary is equivalent for local research: an agent receives only
public frames/actions and its own memory. Before competition-style swarm
submission, wrap this interface with an adapter rather than changing the research
agents' internal learning loop.

Local episodes start with `env.reset()` because the local wrapper exposes reset as
an environment operation. If a game later exposes `RESET` (`0`) during play, the
runner treats it as an ordinary legal public action and counts it like any other
action.

This is a structural research boundary, not a security sandbox. Agent modules
must not import or retain the environment, inspect `env._game`, or read game
implementation files. Oracle diagnostics belong outside the contestant process
and must be reported as oracle-qualified analysis.
