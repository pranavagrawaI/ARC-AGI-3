# Executable LS20 world model

This workspace contains an exact executable copy of the LS20 game source and the engine
transition/rendering machinery it uses. The only substituted component is a small record-
type compatibility shim replacing Pydantic, which the sandbox does not expose; it does not
change LS20 state transitions, collisions, level data, counters, or rendering.

Start a fresh simulated game and try an action sequence with:

```python
world = {}
exec(open("ls20_world.py").read(), world)
result = world["simulate"]([1, 4, 4])
print(result["levels_completed"], result["state"])
print(result["frame"])
```

Action IDs have the same meanings and effects as the live game. Every `simulate` call starts
from a fresh level-1 reset, so you can evaluate alternative sequences without consuming live
actions. The returned `frame` is the final 64x64 integer grid; `frames` contains the final
grid after every simulated action. A sequence that returns `levels_completed >= 1` beats
the probe. Use the simulator for planning, then send the chosen actions to the live game.

The sandbox intentionally does not import model-authored files as modules. Therefore
`ls20_world.py` is a mechanically concatenated, directly executable bundle containing the
engine classes, exact obfuscated game implementation, and a small `simulate` facade. The
same exact game implementation is also present separately as `ls20.py`; `arcengine/` holds
the corresponding engine sources for inspection.
