## H1: A hidden/invisible player moves with arrows (e.g. inside B white block or under bg)
status: untested | test: single UP/LEFT probes, watch diff_regions
## H2: Arrow keys shift whole objects (e.g. white block B or F fill)
status: untested | test: same probes; F-fill or B moving => H2
## H3: Click-based game: arrows inert, CLICK targets do things
status: untested | test: if arrows all no_impact, probe CLICK on F gap / G
## H4: Goal = fill/transform one region (make G white? close F's gap? make D's inside black to match C?)
status: untested | test: level_terminal evidence after a win

## H1: invisible player moves with arrows
status: weakened | a bg-colored player would show on non-5 surfaces; none seen; UP changed only (63,63)
## H2: arrows shift whole objects
status: contradicted (for UP) | UP moved no element
## H5: (63,63) is a step counter (4 + n_actions), arrows otherwise no-ops
status: untested->testing | LEFT,RIGHT batch: expect (63,63) 5->6->7
## H3: click-driven game (CLICK/SPACEBAR do the real work)
status: untested | probe after counter confirmed

