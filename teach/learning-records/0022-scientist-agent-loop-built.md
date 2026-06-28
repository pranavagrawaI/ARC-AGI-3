# Scientist agent loop learns a live depth-2 law by revisiting shapes

Lesson 0014 stacks 0008→0013 into `choose_action` + `run_ls20_agent`: pick_probe
scores cell-ahead simulated contacts, adds navigation toward the target shape when
the floor is ahead, and records one trusted contact event per step. On ls20 (400
steps, seed 17) vs the 0013 random audit: 76 contacts on 8 shapes with
revisit_ratio=1.0 (every shape revisited), 4 actor-change effects, and
`learn_clause` returns LEARNED `IF actor_color_* AND target_shape_* THEN effect`
where random walk hit GRAMMAR_GAP. The law is early/fragile but proves the stack
closes on live frames. Next: goal induction and planning over the induced model.
