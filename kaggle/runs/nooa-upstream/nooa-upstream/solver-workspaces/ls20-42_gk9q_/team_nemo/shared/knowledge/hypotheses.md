# Hypotheses
| id | hypothesis | status | evidence / deciding test |
|----|-----------|--------|--------------------------|
| H1 | white dot at (32,21) is the player, moved by arrows | untested | submit UP/RIGHT, see which cells move |
| H2 | goal: match blue C emblem to head OR reach head | untested | move toward head, watch yellow bar / level_completed |
| H3 | red blocks bottom-right = 4 lives | untested | take damage? (watch for block loss) |
| H4 | gray is solid; cavity rows30-39/cols29-33 is walkable | untested | walk into cavity |

| H5 | player = orange/blue 5x5 token; arrows move it 5 cells; anchored to start (45,34) | confirmed | RIGHT,UP -> (40,39) |
| H6 | goal = deliver token to head at top-left (10,35) | untested | path token up the body into head, watch level_completed |
| H7 | cavity rows30-39/cols29-33 is an obstacle the token cannot enter (or a required waypoint) | untested | try to move token into it |
| H8 | yellow bar = remaining moves (~21), 2/move | untested | count moves vs width; see what happens at 0 |

