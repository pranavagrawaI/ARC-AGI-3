# LS20 tutorial

Your objective is to complete the current level. The game contains seven levels, but this
probe ends successfully as soon as you complete level 1.

## Controls

- Action 1: move up one grid cell.
- Action 2: move down one grid cell.
- Action 3: move left one grid cell.
- Action 4: move right one grid cell.

The player is the two-colour 5x5 rectangular object. One action normally moves it by five
screen pixels, which is one map cell. Solid grey 5x5 blocks are walls. A move into a wall
does not change the player's position, but it still consumes move budget.

## Objective and goals

The player has three attributes: shape, colour, and orientation. The small icon at the
bottom-left of the screen displays the player's current attributes.

A goal is a grey 5x5 square inside a grey 7x7 outline. A small patterned icon associated
with the goal shows the exact shape, colour, and orientation required by that goal. To
complete a goal, make the player's three attributes match that icon and move onto the grey
5x5 goal square. Entering a goal with the wrong attributes is blocked and briefly signals
failure. A completed goal and its icon disappear. The level is complete when every goal in
the level has been completed. Level 1 has one goal.

## Attribute changers

Moving onto these non-blocking 5x5 objects changes one player attribute:

- Shape changer: a sparse black shape-symbol. It advances the player's shape to the next
  shape in a fixed six-shape cycle.
- Colour changer: a four-colour square surrounding a black centre. It advances the
  player's colour through a fixed four-colour cycle.
- Rotation changer: a small black/blue turning-arrow symbol. It rotates the player 90
  degrees clockwise through the cycle 0, 90, 180, 270 degrees.

The other two attributes are preserved. You may revisit a changer to cycle repeatedly.
Some later levels put changers on invisible tracks, causing them to move automatically once
per player action; their effect still occurs only when the player enters their current cell.

## Move budget and lives

The horizontal bar along rows 61-62, beginning near column 13, is remaining move budget.
It is a constraint, not progress and not an objective. Ordinary actions reduce it. Emptying
it does not complete anything. Attempting to move after the budget is exhausted costs one
of the three lives shown at the lower right and resets the current level, including player
position, attributes, goals, changers, and pickups. Losing all three lives ends the game.

A small hollow grey 3x3 square is a refill pickup. Entering it removes the pickup and
restores the current level's move budget to full. Collect refills when the route requires
more moves; reducing the bar to zero is never useful by itself.

## Other mechanics

- Thin blue edge-bars are push/transport devices. Contact pushes the device along its
  direction to the next anchor while carrying the player, then the device returns. The
  carried displacement is part of the same environment action.
- The final level uses fog: cells farther than about four grid cells from the player are
  hidden. The current-attribute icon remains visible.
- A level transition is reported by an increase in `levels_completed`. For this probe,
  `levels_completed >= 1` is success. Conserve actions and plan before moving.

This file describes mechanics, not a route. Derive the level-1 route from the current frame.
