# LR-0016 — The Closed Alphabet

Core Knowledge gives the agent an **observational alphabet, not a victory
language**. The agent may know how to compare, track, and describe change. It may
not know what any game-specific object *means*.

The doctrine:

> Hand-build the alphabet — slots, comparison lenses, identity tracking.
> The agent writes clauses over that alphabet.
> Only a failed clause-space earns a new letter.

## GIVEN (the prior line — frozen)

- objectness
- agentness
- spatial relations (adjacency, containment, alignment)
- time / order
- equality / comparison
- counting
- basic attributes (colour, shape, area, position)
- candidate invariances (translation, rotation, reflection, colour-relabel)
- object persistence

## LEARNED

- game-specific **clauses** over those primitives, induced by interaction.

## FORBIDDEN (rules that would let the agent solve ls20 without playing)

- hardcoded object roles: `door`, `key`, `switch`, `goal`, `cycler`, `portal`
- hardcoded win-condition concepts: `match-symbol-to-win`, `open/close`
- `tile at coordinate causes X`, except as a *learned* ground clause from
  interaction

## Acceptance test

> If a rule would let the agent solve ls20 **without interaction**, it is forbidden.

## What this commits us to

The next system is **not** "predicate invention." It is **bounded clause search
over fixed primitives, fixed slots, multiple comparison lenses, and uncertain
object identity**. Slot *discovery* (inventing a new letter) is deferred until a
high-confidence slot-gap report proves the closed alphabet is insufficient —
first we need a corpse.
