# Lesson 0013 picks probes by guaranteed hypothesis kills per action

Pranav directed the curriculum toward minimal pipeline bricks before assembling a
full agent. Lesson 0013 adds `record_interaction_event` (facts + actor body change
→ `Event`) and `pick_probe`, which scores agency-grounded probes by maximin clause
kills divided by action cost over an explicit version space built from trusted
events plus the probe-menu alphabet. Six synthetic checks pass; a greedy loop
learns `contact AND target_shape=cross` in four probes versus eight for a fixed
weak probe. Reference: `teach/reference/info-gain-policy.html`.

Live audit (`--live`, 400 steps, seed 17): 0008 finds controlled signatures
`(9,15)` and `(12,10)`; pipeline emits 272 trusted contact events (one per step
max, aligned/near spam removed); one `effect=True` via appearance diff when
tracker missed `CHANGED_COLOR`; `learn_clause` returns `GRAMMAR_GAP` because
many distinct target-shape literals contradict at depth 2 — expected until the
agent deliberately revisits shapes. Next: stack into one agent loop with
info-gain movement toward discriminating contacts.
